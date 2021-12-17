pub fn compare(string: &str, other: &str, max_distance: i64) -> Option<i64> {
    use triple_accel::levenshtein::{levenshtein_simd_k};
    let dist = levenshtein_simd_k(
        string.as_ref(),
        other.as_ref(),
        max_distance as u32,
    );

    dist.map(|v| v as i64)
}