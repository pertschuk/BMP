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

#[derive(Serialize, Deserialize, Clone)]
pub enum Postings {
    Sparse(Vec<(u8, u8)>),
    Dense(Vec<u8>),
}

#[derive(Default, Serialize, Deserialize, Clone)]
pub struct BlockForwardIndex {
    pub data: Vec<Vec<(u16, Postings)>>,
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

pub fn fwd2bfwd_with_dense_ratio(fwd: &ForwardIndex, block_size: usize, dense_ratio: f32) -> BlockForwardIndex {
    // Step 1: Group documents into blocks
    let blocks = fwd.data.par_chunks(block_size);
    let progress = indicatif::ProgressBar::new(blocks.len() as u64);
    progress.set_style(pb_style());
    progress.set_draw_delta((blocks.len() / 100) as u64);

    // Clamp ratio between 0.0 and 1.0
    let dense_ratio = dense_ratio.clamp(0.0, 1.0);

    // Step 2: For each block, aggregate term-score pairs
    let data = blocks
        .map(|block| {

            let mut term_pairs: Vec<(u32, u32, u32)> = block.iter().enumerate().flat_map(|(idx, doc)| {
                doc.iter().map(move|(term, score)| (*term, idx as u32, *score))
            }).collect();
            // Sort by term to aggregate them in the next step
            term_pairs.sort_by_key(|pair| pair.0);

            // Aggregate term-score pairs in sparse format first
            let mut aggregated_sparse: Vec<(u16, Vec<(u8, u8)>)> = Vec::new();
            let mut current_term = None;
            let mut current_scores = Vec::new();
            for (term,doc_id, score) in term_pairs {
                match current_term {
                    Some(t) if t == term => current_scores.push((
                        doc_id as u8,
                        score as u8,
                    )),
                    _ => {
                        if let Some(t) = current_term {
                            aggregated_sparse.push((t as u16, current_scores.clone()));
                            current_scores.clear();
                        }
                        current_term = Some(term);
                        current_scores.push((doc_id as u8,score as u8));
                    }
                }
            }
            if let Some(t) = current_term {
                aggregated_sparse.push((t as u16, current_scores));
            }
            progress.inc(1);

            // Decide sparse vs dense per term
            let threshold = (block_size as f32 * dense_ratio).ceil() as usize;
            let mut out: Vec<(u16, Postings)> = Vec::with_capacity(aggregated_sparse.len());
            for (term, pairs) in aggregated_sparse.into_iter() {
                if pairs.len() > threshold {
                    // Dense: expand into a length==block_size vector of u8 scores initialized to 0
                    let mut dense = vec![0u8; block_size];
                    for (doc_id, score) in pairs.into_iter() {
                        dense[doc_id as usize] = score;
                    }
                    out.push((term, Postings::Dense(dense)));
                } else {
                    out.push((term, Postings::Sparse(pairs)));
                }
            }

            out
        })
        .collect();

    BlockForwardIndex { block_size, data }
}

pub fn fwd2bfwd(fwd: &ForwardIndex, block_size: usize) -> BlockForwardIndex {
    // default: dense if more than half of docs are present
    fwd2bfwd_with_dense_ratio(fwd, block_size, 0.25)
}

#[cfg(not(all(target_arch = "x86_64", feature = "avx512f", feature = "avx512bw")))]
#[inline]
pub fn block_score(
    query: &Vec<(u16, u8)>,
    document: &[(u16, Postings)],
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
                match &(*term_ptr).1 {
                    Postings::Sparse(pairs) => {
                        let mut inner_ptr = pairs.as_ptr();
                        let end_inner_ptr = inner_ptr.wrapping_offset(pairs.len() as isize);
                        while inner_ptr != end_inner_ptr {
                            doc_score[(*inner_ptr).0 as usize] += (value as u16) * ((*inner_ptr).1 as u16);
                            inner_ptr = inner_ptr.add(1);
                        }
                    }
                    Postings::Dense(dense) => {
                        // dense vector aligned to block_size, each position is a score
                        let mut inner_ptr = dense.as_ptr();
                        let end_inner_ptr = inner_ptr.wrapping_offset(dense.len() as isize);
                        let mut idx: usize = 0;
                        while inner_ptr != end_inner_ptr {
                            let s = *inner_ptr as u16;
                            if s != 0 {
                                doc_score[idx] += (value as u16) * s;
                            }
                            inner_ptr = inner_ptr.add(1);
                            idx += 1;
                        }
                    }
                }
            }
        }
    }
    doc_score
}


#[cfg(all(target_arch = "x86_64", feature = "avx512f", feature = "avx512bw"))]
#[inline]
pub fn block_score(
    query: &Vec<(u16, u8)>,
    document: &[(u16, Postings)],
    bsize: usize,
) -> Vec<u16> {
    assert_eq!(bsize, 32);
    let mut doc_score = vec![0u16; bsize];
    let dense_acc = _mm512_setzero_si512();

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
                match &(*term_ptr).1 {
                    Postings::Sparse(pairs) => {
                        let mut inner_ptr = pairs.as_ptr();
                        let end_inner_ptr = inner_ptr.wrapping_offset(pairs.len() as isize);
                        while inner_ptr != end_inner_ptr {
                            doc_score[(*inner_ptr).0 as usize] += (value as u16) * ((*inner_ptr).1 as u16);
                            inner_ptr = inner_ptr.add(1);
                        }
                    }
                    Postings::Dense(dense) => {
                        // dense vector aligned to block_size, each position is a score
                        let scores = _mm512_loadu_si512(dense.as_ptr() as *const __m512i);
                        let uint16_scores = _mm512_cvtepi8_epi16(scores);
                        let scores = _mm512_mullo_epi16(scores, _mm512_set1_epi16(value as i16));
                        let dense_acc = _mm512_add_epi16(scores, dense_acc);
                    }
                }
            }
        }
    }
    let sparse_scores = _mm512_loadu_si512(doc_score.as_ptr() as *const __m512i);
    let final_scores = _mm512_add_epi16(sparse_scores, dense_acc);
    _mm512_storeu_si512(doc_score.as_mut_ptr() as *mut _, final_scores);
    doc_score
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_block_score() {
        // Test mixed and sparse Postings Lists
        // Test with mixed (Sparse and Dense) Postings
        let query = vec![(1, 2), (2, 3)];
        let document = vec![
            (1, Postings::Sparse(vec![(0, 1), (2, 2)])),
            (2, Postings::Dense(vec![1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ];
        let bsize = 32;
        let doc_score = block_score(&query, &document, bsize);
        // For coordinate 1 (Sparse): (0,1) and (2,2) with value 2
        // doc_score[0] += 2*1 = 2
        // doc_score[2] += 2*2 = 4
        // For coordinate 2 (Dense): [1,2,3,4,...] with value 3
        // doc_score[0] += 3*1 = 3
        // doc_score[1] += 3*2 = 6
        // doc_score[2] += 3*3 = 9
        // doc_score[3] += 3*4 = 12
        // So final doc_score[0] = 2+3=5, [1]=6, [2]=4+9=13, [3]=12
        let mut expected = vec![0u16; 32];
        expected[0] = 5;
        expected[1] = 6;
        expected[2] = 13;
        expected[3] = 12;
        assert_eq!(doc_score, expected);

        // Test with only Sparse Postings
        let query = vec![(5, 1)];
        let document = vec![
            (5, Postings::Sparse(vec![(0, 2), (3, 4), (7, 1)])),
        ];
        let bsize = 8;
        let doc_score = block_score(&query, &document, bsize);
        let mut expected = vec![0u16; 8];
        expected[0] = 2;
        expected[3] = 4;
        expected[7] = 1;
        assert_eq!(doc_score, expected);

        // Test with only Dense Postings
        let query = vec![(10, 2)];
        let document = vec![
            (10, Postings::Dense(vec![1, 0, 2, 3, 0, 0, 0, 0])),
        ];
        let bsize = 8;
        let doc_score = block_score(&query, &document, bsize);
        let mut expected = vec![0u16; 8];
        expected[0] = 2;
        expected[2] = 4;
        expected[3] = 6;
        assert_eq!(doc_score, expected);
    }
}