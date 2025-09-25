use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{prelude::*, seq::SliceRandom};

use bmp::index::forward_index::{block_score, Postings};

#[derive(Clone)]
struct BenchDoc {
    query: Vec<(u16, u8)>,
    document: Vec<(u16, Postings)>,
    block_size: usize,
}

fn gen_bench_doc(
    num_terms: usize,
    query_len: usize,
    block_size: usize,
    dense_ratio: f32, // fraction of terms that will be dense
    avg_sparse_len: usize,
) -> BenchDoc {
    let mut rng = StdRng::seed_from_u64(12345);
    let mut all_terms: Vec<u16> = (0..num_terms).map(|i| i as u16).collect();
    all_terms.shuffle(&mut rng);
    let query_terms = all_terms.iter().copied().take(query_len).collect::<Vec<_>>();

    let query = query_terms
        .iter()
        .map(|&t| (t, rng.gen_range(1..=255)))
        .collect::<Vec<_>>();

    let num_dense = ((num_terms as f32) * dense_ratio).round() as usize;
    let dense_terms: std::collections::HashSet<u16> = all_terms
        .iter()
        .copied()
        .take(num_dense)
        .collect();

    let mut document: Vec<(u16, Postings)> = Vec::with_capacity(num_terms);
    for term in 0..num_terms {
        let term_id = term as u16;
        if dense_terms.contains(&term_id) {
            let mut dense = vec![0u8; block_size];
            for i in 0..block_size {
                // 50% chance populated, random small scores
                if rng.gen_bool(0.5) {
                    dense[i] = rng.gen_range(1..=5);
                }
            }
            document.push((term_id, Postings::Dense(dense)));
        } else {
            let mut pairs = Vec::new();
            let len = rng.gen_range(0..=avg_sparse_len * 2);
            for _ in 0..len {
                let did = rng.gen_range(0..block_size) as u8;
                let sc = rng.gen_range(1..=5) as u8;
                pairs.push((did, sc));
            }
            // Sorting improves the scalar walk in block_score
            pairs.sort_unstable_by_key(|p| p.0);
            document.push((term_id, Postings::Sparse(pairs)));
        }
    }

    // Ensure document is sorted by term id as expected by block_score's two-pointer walk
    document.sort_unstable_by_key(|(t, _)| *t);

    BenchDoc {
        query,
        document,
        block_size,
    }
}

fn bench_block_score_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_score_ratios");
    let block_size = 32;
    let num_terms = 200;
    let query_len = 10;
    let avg_sparse_len = 8;

    for &dense_ratio in &[0.0_f32, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let data = gen_bench_doc(num_terms, query_len, block_size, dense_ratio, avg_sparse_len);
        group.throughput(Throughput::Elements(block_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("dense_{:.2}", dense_ratio)),
            &data,
            |b, d| {
                b.iter(|| block_score(black_box(&d.query), black_box(&d.document), black_box(d.block_size)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_block_score_ratios);
criterion_main!(benches);


