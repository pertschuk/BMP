use indicatif::ProgressStyle;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const DEFAULT_PROGRESS_TEMPLATE: &str =
    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {count}/{total} ({eta})";

/// Returns default progress style.
fn pb_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(DEFAULT_PROGRESS_TEMPLATE)
        .progress_chars("=> ")
}

#[derive(Default, Serialize, Deserialize, Clone)]
pub struct BlockDocument {
    pub terms: Vec<u16>,
    pub docs_impacts: Vec<Vec<(u8, u8)>>,
}

#[derive(Default, Serialize, Deserialize, Clone)]
pub struct BlockForwardIndex {
    pub data: Vec<Vec<(u16, (Vec<u8>, Vec<u8>))>>,
    pub block_size: usize,
}

#[derive(Default, Serialize, Deserialize, Clone)]
pub struct ForwardIndex {
    data: Vec<Vec<(u32, u32)>>,
}
pub struct ForwardIndexBuilder {
    forward_index: ForwardIndex,
}
// Implement IntoIterator for a reference to PostingList.
impl<'a> IntoIterator for &'a ForwardIndex {
    type Item = &'a Vec<(u32, u32)>;
    type IntoIter = std::slice::Iter<'a, Vec<(u32, u32)>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl ForwardIndexBuilder {
    pub fn new(num_documents: usize) -> ForwardIndexBuilder {
        Self {
            forward_index: ForwardIndex {
                data: vec![Vec::new(); num_documents],
            },
        }
    }
    pub fn insert_posting_list(&mut self, term_id: u32, posting_list: &Vec<(u32, u32)>) {
        for (doc_id, score) in posting_list {
            self.forward_index.data[*doc_id as usize].push((term_id as u32, *score));
        }
    }
    pub fn insert_document(&mut self, vector: Vec<(u32, u32)>) {
        self.forward_index.data.push(vector);
    }
    pub fn build(&mut self) -> ForwardIndex {
        for doc in &mut self.forward_index.data {
            doc.sort_by_key(|d| d.0);
        }

        std::mem::take(&mut self.forward_index)
    }
}

pub fn fwd2bfwd(fwd: &ForwardIndex, block_size: usize) -> BlockForwardIndex {
    // Step 1: Group documents into blocks
    let blocks = fwd.data.par_chunks(block_size);
    let progress = indicatif::ProgressBar::new(blocks.len() as u64);
    progress.set_style(pb_style());
    progress.set_draw_delta((blocks.len() / 100) as u64);

    BlockForwardIndex {
        block_size: block_size,
        data: blocks
        .map(|block| {

            let mut term_pairs: Vec<(u32, u32, u32)> = block.iter().enumerate().flat_map(|(idx, doc)| {
                doc.iter().map(move|(term, score)| (*term, idx as u32, *score))
            }).collect();
            // Sort by term to aggregate them in the next step
            term_pairs.sort_by_key(|pair| pair.0);

            // Aggregate term-score pairs
            let mut aggregated: Vec<(u16, (Vec<u8>, Vec<u8>))> = Vec::new();
            let mut current_term = None;
            let mut current_doc_ids = Vec::new();
            let mut current_scores = Vec::new();
            for (term,doc_id, score) in term_pairs {
                match current_term {
                    Some(t) if t == term => {
                        current_doc_ids.push(doc_id as u8);
                        current_scores.push(score as u8);
                    },
                    _ => {
                        if let Some(t) = current_term {
                            aggregated.push((t as u16, (current_doc_ids.clone(), current_scores.clone())));
                            current_doc_ids.clear();
                            current_scores.clear();
                        }
                        current_term = Some(term);
                        current_doc_ids.push(doc_id as u8);
                        current_scores.push(score as u8);
                    }
                }
            }
            if let Some(t) = current_term {
                aggregated.push((t as u16, (current_doc_ids, current_scores)));
            }
            progress.inc(1);

            aggregated
        })
        .collect()
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
#[inline]
pub fn block_score(
    query: &[(u16, u8)],
    document: &[(u16, (Vec<u8>, Vec<u8>))],
    bsize: usize,
) -> Vec<u16> {
    use std::arch::x86_64::*;
    assert!(bsize == 16);

    unsafe {
        // In-register accumulator: 16 lanes of i32, one per doc id 0..15
        let mut acc: __m512i = _mm512_setzero_si512();
        // Lane indices [0..15]
        let idx: __m512i = _mm512_set_epi32(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        );

        let mut term_ptr = document.as_ptr();
        let end = term_ptr.wrapping_offset(document.len() as isize);

        for &(coordinate, value) in query {
            while term_ptr != end && (*term_ptr).0 < coordinate {
                term_ptr = term_ptr.add(1);
            }
            if term_ptr == end {
                break;
            }

            if (*term_ptr).0 == coordinate {
                let doc_ids = &(*term_ptr).1.0;
                let scores = &(*term_ptr).1.1;
                let len = doc_ids.len();

                // Active lanes mask (lowest len bits)
                let k_mask: __mmask16 = if len >= 16 { 0xFFFF } else { ((1u32 << len) - 1) as __mmask16 };

                // Masked 16B loads of u8 doc_ids and scores (inactive lanes = 0)
                let docs128: __m128i = _mm_maskz_loadu_epi8(k_mask, doc_ids.as_ptr() as *const _);
                let sc128: __m128i = _mm_maskz_loadu_epi8(k_mask, scores.as_ptr() as *const _);

                // Widen u8 -> i32 (16 lanes)
                let docs_i32: __m512i = _mm512_cvtepu8_epi32(docs128);
                let sc_i32: __m512i = _mm512_cvtepu8_epi32(sc128);

                // Broadcast query weight and precompute contributions per posting lane
                let qv: __m512i = _mm512_set1_epi32(value as i32);
                let contrib: __m512i = _mm512_mullo_epi32(sc_i32, qv);

                // For each active posting lane j, add contrib to lane == doc_id[j]
                // This keeps accumulation entirely in registers.
                // Unroll the loop to avoid using variables in intrinsic function calls
            //     if len > 0 {
            //         let docs_chunk_0: __m128i = _mm512_extracti32x4_epi32(docs_i32, 0);
            //         let sc_chunk_0: __m128i = _mm512_extracti32x4_epi32(contrib, 0);
                    
            //         if len > 0 {
            //             let doc_id_0: i32 = _mm_extract_epi32(docs_chunk_0, 0);
            //             let val_0: i32 = _mm_extract_epi32(sc_chunk_0, 0);
            //             let doc_bcast_0: __m512i = _mm512_set1_epi32(doc_id_0);
            //             let m_0: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_0);
            //             let val_bcast_0: __m512i = _mm512_set1_epi32(val_0);
            //             acc = _mm512_mask_add_epi32(acc, m_0, acc, val_bcast_0);
            //         }
            //         if len > 1 {
            //             let doc_id_1: i32 = _mm_extract_epi32(docs_chunk_0, 1);
            //             let val_1: i32 = _mm_extract_epi32(sc_chunk_0, 1);
            //             let doc_bcast_1: __m512i = _mm512_set1_epi32(doc_id_1);
            //             let m_1: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_1);
            //             let val_bcast_1: __m512i = _mm512_set1_epi32(val_1);
            //             acc = _mm512_mask_add_epi32(acc, m_1, acc, val_bcast_1);
            //         }
            //         if len > 2 {
            //             let doc_id_2: i32 = _mm_extract_epi32(docs_chunk_0, 2);
            //             let val_2: i32 = _mm_extract_epi32(sc_chunk_0, 2);
            //             let doc_bcast_2: __m512i = _mm512_set1_epi32(doc_id_2);
            //             let m_2: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_2);
            //             let val_bcast_2: __m512i = _mm512_set1_epi32(val_2);
            //             acc = _mm512_mask_add_epi32(acc, m_2, acc, val_bcast_2);
            //         }
            //         if len > 3 {
            //             let doc_id_3: i32 = _mm_extract_epi32(docs_chunk_0, 3);
            //             let val_3: i32 = _mm_extract_epi32(sc_chunk_0, 3);
            //             let doc_bcast_3: __m512i = _mm512_set1_epi32(doc_id_3);
            //             let m_3: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_3);
            //             let val_bcast_3: __m512i = _mm512_set1_epi32(val_3);
            //             acc = _mm512_mask_add_epi32(acc, m_3, acc, val_bcast_3);
            //         }
            //     }
                
            //     if len > 4 {
            //         let docs_chunk_1: __m128i = _mm512_extracti32x4_epi32(docs_i32, 1);
            //         let sc_chunk_1: __m128i = _mm512_extracti32x4_epi32(contrib, 1);
                    
            //         if len > 4 {
            //             let doc_id_4: i32 = _mm_extract_epi32(docs_chunk_1, 0);
            //             let val_4: i32 = _mm_extract_epi32(sc_chunk_1, 0);
            //             let doc_bcast_4: __m512i = _mm512_set1_epi32(doc_id_4);
            //             let m_4: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_4);
            //             let val_bcast_4: __m512i = _mm512_set1_epi32(val_4);
            //             acc = _mm512_mask_add_epi32(acc, m_4, acc, val_bcast_4);
            //         }
            //         if len > 5 {
            //             let doc_id_5: i32 = _mm_extract_epi32(docs_chunk_1, 1);
            //             let val_5: i32 = _mm_extract_epi32(sc_chunk_1, 1);
            //             let doc_bcast_5: __m512i = _mm512_set1_epi32(doc_id_5);
            //             let m_5: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_5);
            //             let val_bcast_5: __m512i = _mm512_set1_epi32(val_5);
            //             acc = _mm512_mask_add_epi32(acc, m_5, acc, val_bcast_5);
            //         }
            //         if len > 6 {
            //             let doc_id_6: i32 = _mm_extract_epi32(docs_chunk_1, 2);
            //             let val_6: i32 = _mm_extract_epi32(sc_chunk_1, 2);
            //             let doc_bcast_6: __m512i = _mm512_set1_epi32(doc_id_6);
            //             let m_6: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_6);
            //             let val_bcast_6: __m512i = _mm512_set1_epi32(val_6);
            //             acc = _mm512_mask_add_epi32(acc, m_6, acc, val_bcast_6);
            //         }
            //         if len > 7 {
            //             let doc_id_7: i32 = _mm_extract_epi32(docs_chunk_1, 3);
            //             let val_7: i32 = _mm_extract_epi32(sc_chunk_1, 3);
            //             let doc_bcast_7: __m512i = _mm512_set1_epi32(doc_id_7);
            //             let m_7: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_7);
            //             let val_bcast_7: __m512i = _mm512_set1_epi32(val_7);
            //             acc = _mm512_mask_add_epi32(acc, m_7, acc, val_bcast_7);
            //         }
            //     }
                
            //     if len > 8 {
            //         let docs_chunk_2: __m128i = _mm512_extracti32x4_epi32(docs_i32, 2);
            //         let sc_chunk_2: __m128i = _mm512_extracti32x4_epi32(contrib, 2);
                    
            //         if len > 8 {
            //             let doc_id_8: i32 = _mm_extract_epi32(docs_chunk_2, 0);
            //             let val_8: i32 = _mm_extract_epi32(sc_chunk_2, 0);
            //             let doc_bcast_8: __m512i = _mm512_set1_epi32(doc_id_8);
            //             let m_8: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_8);
            //             let val_bcast_8: __m512i = _mm512_set1_epi32(val_8);
            //             acc = _mm512_mask_add_epi32(acc, m_8, acc, val_bcast_8);
            //         }
            //         if len > 9 {
            //             let doc_id_9: i32 = _mm_extract_epi32(docs_chunk_2, 1);
            //             let val_9: i32 = _mm_extract_epi32(sc_chunk_2, 1);
            //             let doc_bcast_9: __m512i = _mm512_set1_epi32(doc_id_9);
            //             let m_9: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_9);
            //             let val_bcast_9: __m512i = _mm512_set1_epi32(val_9);
            //             acc = _mm512_mask_add_epi32(acc, m_9, acc, val_bcast_9);
            //         }
            //         if len > 10 {
            //             let doc_id_10: i32 = _mm_extract_epi32(docs_chunk_2, 2);
            //             let val_10: i32 = _mm_extract_epi32(sc_chunk_2, 2);
            //             let doc_bcast_10: __m512i = _mm512_set1_epi32(doc_id_10);
            //             let m_10: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_10);
            //             let val_bcast_10: __m512i = _mm512_set1_epi32(val_10);
            //             acc = _mm512_mask_add_epi32(acc, m_10, acc, val_bcast_10);
            //         }
            //         if len > 11 {
            //             let doc_id_11: i32 = _mm_extract_epi32(docs_chunk_2, 3);
            //             let val_11: i32 = _mm_extract_epi32(sc_chunk_2, 3);
            //             let doc_bcast_11: __m512i = _mm512_set1_epi32(doc_id_11);
            //             let m_11: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_11);
            //             let val_bcast_11: __m512i = _mm512_set1_epi32(val_11);
            //             acc = _mm512_mask_add_epi32(acc, m_11, acc, val_bcast_11);
            //         }
            //     }
                
            //     if len > 12 {
            //         let docs_chunk_3: __m128i = _mm512_extracti32x4_epi32(docs_i32, 3);
            //         let sc_chunk_3: __m128i = _mm512_extracti32x4_epi32(contrib, 3);
                    
            //         if len > 12 {
            //             let doc_id_12: i32 = _mm_extract_epi32(docs_chunk_3, 0);
            //             let val_12: i32 = _mm_extract_epi32(sc_chunk_3, 0);
            //             let doc_bcast_12: __m512i = _mm512_set1_epi32(doc_id_12);
            //             let m_12: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_12);
            //             let val_bcast_12: __m512i = _mm512_set1_epi32(val_12);
            //             acc = _mm512_mask_add_epi32(acc, m_12, acc, val_bcast_12);
            //         }
            //         if len > 13 {
            //             let doc_id_13: i32 = _mm_extract_epi32(docs_chunk_3, 1);
            //             let val_13: i32 = _mm_extract_epi32(sc_chunk_3, 1);
            //             let doc_bcast_13: __m512i = _mm512_set1_epi32(doc_id_13);
            //             let m_13: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_13);
            //             let val_bcast_13: __m512i = _mm512_set1_epi32(val_13);
            //             acc = _mm512_mask_add_epi32(acc, m_13, acc, val_bcast_13);
            //         }
            //         if len > 14 {
            //             let doc_id_14: i32 = _mm_extract_epi32(docs_chunk_3, 2);
            //             let val_14: i32 = _mm_extract_epi32(sc_chunk_3, 2);
            //             let doc_bcast_14: __m512i = _mm512_set1_epi32(doc_id_14);
            //             let m_14: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_14);
            //             let val_bcast_14: __m512i = _mm512_set1_epi32(val_14);
            //             acc = _mm512_mask_add_epi32(acc, m_14, acc, val_bcast_14);
            //         }
            //         if len > 15 {
            //             let doc_id_15: i32 = _mm_extract_epi32(docs_chunk_3, 3);
            //             let val_15: i32 = _mm_extract_epi32(sc_chunk_3, 3);
            //             let doc_bcast_15: __m512i = _mm512_set1_epi32(doc_id_15);
            //             let m_15: __mmask16 = _mm512_cmpeq_epi32_mask(idx, doc_bcast_15);
            //             let val_bcast_15: __m512i = _mm512_set1_epi32(val_15);
            //             acc = _mm512_mask_add_epi32(acc, m_15, acc, val_bcast_15);
            //         }
            //     }
            }
        }

        // Store accumulator and convert to u16
        let mut out = vec![0i32; 16];
        _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
        out.into_iter().map(|x| x as u16).collect()
    }
}

#[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
#[inline]
pub fn block_score(
    query: &[(u16, u8)],
    document: &[(u16, (Vec<u8>, Vec<u8>))],
    bsize: usize,
) -> Vec<u16> {
    let mut doc_score = vec![0; bsize];

    unsafe {
        let mut term_ptr = document.as_ptr();
        let end = term_ptr.wrapping_offset(document.len() as isize);
        for &(coordinate, value) in query {
            while term_ptr != end && (*term_ptr).0 < coordinate {
                term_ptr = term_ptr.add(1);
            }
            if term_ptr == end {
                break;
            }
            if (*term_ptr).0 == coordinate {
                let (doc_ids, scores) = &(*term_ptr).1;
                let doc_ids_ptr = doc_ids.as_ptr();
                let scores_ptr = scores.as_ptr();
                let len = doc_ids.len();
                
                for i in 0..len {
                    let doc_id = *doc_ids_ptr.add(i) as usize;
                    let score = *scores_ptr.add(i) as u16;
                    doc_score[doc_id] += (value as u16) * score;
                }
            }
        }
    }

    doc_score
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_score_basic() {
        // Query: term 1 with weight 2, term 2 with weight 3
        let query = vec![(1u16, 2u8), (2u16, 3u8)];
        // Document: term 1 appears in doc 0 (score 5), term 2 in doc 1 (score 7)
        let document = vec![
            (1u16, (vec![0u8], vec![5u8])),
            (2u16, (vec![1u8], vec![7u8])),
        ];
        let bsize = 16;
        let result = block_score(&query, &document, bsize);
        // doc 0: 2*5 = 10, doc 1: 3*7 = 21, doc 2..15: 0
        let mut expected = vec![0u16; 16];
        expected[0] = 10;
        expected[1] = 21;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_block_score_multiple_postings() {
        // Query: term 1 with weight 1
        let query = vec![(1u16, 1u8)];
        // Document: term 1 appears in doc 0 (score 2), doc 1 (score 3), doc 2 (score 4)
        let document = vec![
            (1u16, (vec![0u8, 1u8, 2u8], vec![2u8, 3u8, 4u8])),
        ];
        let bsize = 16;
        let result = block_score(&query, &document, bsize);
        let mut expected = vec![0u16; 16];
        expected[0] = 2;
        expected[1] = 3;
        expected[2] = 4;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_block_score_empty_query() {
        let query = vec![];
        let document = vec![
            (1u16, (vec![0u8], vec![5u8])),
        ];
        let bsize = 16;
        let result = block_score(&query, &document, bsize);
        assert_eq!(result, vec![0u16; 16]);
    }

    #[test]
    fn test_block_score_empty_document() {
        let query = vec![(1u16, 2u8)];
        let document: Vec<(u16, (Vec<u8>, Vec<u8>))> = vec![];
        let bsize = 16;
        let result = block_score(&query, &document, bsize);
        assert_eq!(result, vec![0u16; 16]);
    }

    #[test]
    fn test_block_score_multiple_terms_and_docs() {
        // Query: term 1 (weight 2), term 3 (weight 4)
        let query = vec![(1u16, 2u8), (3u16, 4u8)];
        // Document: term 1 in doc 0 (score 1), doc 2 (score 2)
        //           term 2 in doc 1 (score 3)
        //           term 3 in doc 0 (score 2), doc 2 (score 1)
        let document = vec![
            (1u16, (vec![0u8, 2u8], vec![1u8, 2u8])),
            (2u16, (vec![1u8], vec![3u8])),
            (3u16, (vec![0u8, 2u8], vec![2u8, 1u8])),
        ];
        let bsize = 16;
        let result = block_score(&query, &document, bsize);
        // doc 0: 2*1 + 4*2 = 2 + 8 = 10
        // doc 1: 0
        // doc 2: 2*2 + 4*1 = 4 + 4 = 8
        let mut expected = vec![0u16; 16];
        expected[0] = 10;
        expected[2] = 8;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_block_score_unique_doc_ids() {
        // Query: term 1 (weight 2)
        let query = vec![(1u16, 2u8)];
        // Document: term 1 in doc 0 (score 3) and doc 1 (score 4)
        let document = vec![
            (1u16, (vec![0u8, 1u8], vec![3u8, 4u8])),
        ];
        let bsize = 16;
        let result = block_score(&query, &document, bsize);
        // doc 0: 2*3 = 6, doc 1: 2*4 = 8
        let mut expected = vec![0u16; 16];
        expected[0] = 6;
        expected[1] = 8;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_block_score_bsize_larger_than_docs() {
        // Query: term 1 (weight 1)
        let query = vec![(1u16, 1u8)];
        // Document: term 1 in doc 0 (score 5)
        let document = vec![
            (1u16, (vec![0u8], vec![5u8])),
        ];
        let bsize = 16;
        let result = block_score(&query, &document, bsize);
        let mut expected = vec![0u16; 16];
        expected[0] = 5;
        assert_eq!(result, expected);
    }
}
