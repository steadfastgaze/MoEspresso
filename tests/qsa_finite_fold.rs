use half::bf16;
use moespresso::qsa_attention::{
    HEAD_WIDTH, KV_HEADS, QUERY_HEADS, SELECTED_WIDTH, validate_selected_rows,
};

fn folded(parts: [&[u16]; 4]) -> bool {
    parts[0]
        .iter()
        .chain(parts[1])
        .chain(parts[2])
        .chain(parts[3])
        .fold(true, |ok, &word| ok & bf16::from_bits(word).is_finite())
}

fn original(parts: [&[u16]; 4]) -> bool {
    parts[0]
        .iter()
        .chain(parts[1])
        .chain(parts[2])
        .chain(parts[3])
        .all(|&word| bf16::from_bits(word).is_finite())
}

#[test]
fn finite_fold_matches_original_for_every_bf16_encoding_and_slice() {
    assert!(folded([&[], &[], &[], &[]]));
    for word in 0..=u16::MAX {
        let expected = bf16::from_bits(word).is_finite();
        for slice in 0..4 {
            let mut parts = [[0x8000u16, 0x0001, 0x7f7f]; 4];
            parts[slice][word as usize % 3] = word;
            let refs = parts.each_ref().map(|part| part.as_slice());
            assert_eq!(
                folded(refs),
                original(refs),
                "word={word:04x} slice={slice}"
            );
            assert_eq!(folded(refs), expected, "word={word:04x} slice={slice}");
            let one = [word];
            let mut refs: [&[u16]; 4] = [&[], &[], &[], &[]];
            refs[slice] = &one;
            assert_eq!(folded(refs), original(refs));
        }
    }
}

#[test]
fn selected_row_guard_preserves_finite_and_error_contracts() {
    let finite: Vec<_> = (0..=u16::MAX)
        .filter(|&word| bf16::from_bits(word).is_finite())
        .collect();
    let row = |count| {
        finite
            .iter()
            .copied()
            .cycle()
            .take(count)
            .collect::<Vec<_>>()
    };
    let mut parts = [
        row(QUERY_HEADS * HEAD_WIDTH),
        row(QUERY_HEADS * HEAD_WIDTH),
        row(KV_HEADS * SELECTED_WIDTH * HEAD_WIDTH),
        row(KV_HEADS * SELECTED_WIDTH * HEAD_WIDTH),
    ];
    let valid = vec![1u8; SELECTED_WIDTH];
    let check = |parts: &[Vec<u16>; 4], valid: &[u8]| {
        validate_selected_rows(&parts[0], &parts[1], &parts[2], &parts[3], valid)
    };
    check(&parts, &valid).unwrap();
    for slice in 0..4 {
        let last = parts[slice].len() - 1;
        for (position, bad) in [(0, 0x7f80), (last, 0xffff)] {
            let old = parts[slice][position];
            parts[slice][position] = bad;
            assert_eq!(
                check(&parts, &valid).unwrap_err().to_string(),
                "QSA selected inputs must be finite BF16"
            );
            parts[slice][position] = old;
        }
    }
    let mut malformed = valid.clone();
    malformed[1] = 0;
    assert_eq!(
        check(&parts, &malformed).unwrap_err().to_string(),
        "QSA valid rows must precede padding"
    );
    malformed[1] = 2;
    assert_eq!(
        check(&parts, &malformed).unwrap_err().to_string(),
        "QSA validity must contain only zero or one"
    );
    malformed.fill(0);
    assert_eq!(
        check(&parts, &malformed).unwrap_err().to_string(),
        "QSA one-token selection must contain a visible row"
    );
    parts[0].pop();
    assert_eq!(
        check(&parts, &valid).unwrap_err().to_string(),
        "QSA query shape differs from 24x256"
    );
}
