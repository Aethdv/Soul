#![allow(clippy::cast_possible_wrap)]

use std::{env, fs::File, io::Write, path::Path};

type Square = u8;
type Bitboard = u64;

#[derive(Copy, Clone, PartialEq, Eq)]
enum PieceType {
    Rook,
    Bishop,
}

struct MagicEntry {
    mask: u64,
    magic: u64,
    shift: u8,
    offset: u32,
}

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = FILE_A << 7;
const RANK_1: u64 = 0x0000_0000_0000_00FF;

#[rustfmt::skip]
const RANK_MASKS: [u64; 8] = [
    RANK_1,
    RANK_1 << 8,
    RANK_1 << 16,
    RANK_1 << 24,
    RANK_1 << 32,
    RANK_1 << 40,
    RANK_1 << 48,
    RANK_1 << 56,
];

#[rustfmt::skip]
const FILE_MASKS: [u64; 8] = [
    FILE_A,
    FILE_A << 1,
    FILE_A << 2,
    FILE_A << 3,
    FILE_A << 4,
    FILE_A << 5,
    FILE_A << 6,
    FILE_A << 7,
];

// Magic Bitboard Constants (Non-BMI2 only)
// Precomputed magic numbers, generated with seed 0xDEAD_BEEF_CAFE_BABE.
// Only used on non-BMI2 machines for sliding piece attack generation.

#[rustfmt::skip]
const ROOK_MAGICS: [u64; 64] = [
    0x0080_0622_8030_4000, 0x0140_0810_0420_0040, 0x9100_1009_0220_0040, 0x0080_0481_1002_0800,
    0x0080_0800_8014_0003, 0x0100_0900_0400_0802, 0x0280_0200_1100_1880, 0x0500_0224_8200_4700,
    0x0610_8010_4000_2084, 0x0000_2000_0004_0300, 0x2023_0A02_0106_0000, 0x0531_0280_0080_1080,
    0xA0A1_0011_0088_0100, 0x0000_820C_2A58_0490, 0x1000_0220_8021_0200, 0x4000_8268_3004_0080,
    0x0040_0080_0090_4020, 0x0001_2800_4048_1040, 0x0040_0004_0280_1000, 0x0120_4000_0251_8080,
    0x0020_0018_0200_0140, 0x0200_8809_0004_0000, 0x0000_0803_1450_0008, 0x0200_0000_0401_F100,
    0x1228_4008_8000_2980, 0x0641_02A8_4209_0080, 0x1030_0080_8022_0024, 0x0020_0280_3002_2000,
    0xA241_0824_0100_812D, 0x0000_2000_4404_0200, 0x0286_0200_0000_0003, 0x8410_0008_2002_2222,
    0xB000_4000_8080_00E0, 0xC808_4000_0008_8101, 0x2000_1104_0104_0230, 0x0018_6024_0200_2040,
    0x8E00_0001_0000_0310, 0x0200_4800_0009_8000, 0x0100_2200_4000_0001, 0x0008_2001_4048_22A0,
    0x0800_4000_8000_8020, 0x2001_0100_2308_0200, 0x4050_0208_0400_0001, 0xA100_0400_0820_6082,
    0x0003_0208_0040_4400, 0x0040_0004_0800_0000, 0x9020_8008_8000_8026, 0x2000_0200_0080_0000,
    0x0A48_A8C0_0080_0980, 0xE000_1402_0000_0001, 0x0200_4048_0004_0000, 0x040A_0040_3010_EA61,
    0x0020_4080_0000_C150, 0x0016_4000_0008_000A, 0x0800_0000_0090_0002, 0x0000_2804_0028_1000,
    0x2800_6041_0600_1282, 0x8000_2020_8090_4000, 0x00E0_00D0_0000_6120, 0x0008_2000_0012_4040,
    0x8100_2240_0000_0820, 0x0030_0000_0009_000E, 0x2000_0000_8041_4800, 0x0002_4910_2020_2008,
];

#[rustfmt::skip]
const BISHOP_MAGICS: [u64; 64] = [
    0x2190_0108_0080_8202, 0x4011_0242_0400_2004, 0x0008_0801_1160_0020, 0x0008_0E00_C440_0420,
    0x8301_1040_0400_0218, 0x1603_0121_1021_0501, 0x8002_0905_6840_8008, 0x020D_0088_0588_0400,
    0x0020_2020_0302_1480, 0x8000_A810_0C82_00C8, 0x0000_1204_2842_0410, 0x8002_1104_0481_020A,
    0x0008_1404_2000_0801, 0x0802_1208_0208_1A80, 0x8009_0402_A208_2014, 0x0011_002C_0402_0804,
    0x0406_0128_2014_4400, 0x2210_0002_1081_1308, 0x40A0_8050_0028_8260, 0x0184_0048_0411_B000,
    0x4104_808C_00E0_0400, 0x0141_001A_0100_B200, 0x0000_8822_0230_0600, 0x8080_8220_4200_D006,
    0x2024_4004_2490_0480, 0x0008_0910_120A_0825, 0xC000_4C01_0808_0010, 0x0026_0800_0400_4288,
    0x0801_0010_0100_4008, 0x0230_0100_4880_4900, 0x0043_1118_0248_0804, 0x8201_8280_0100_C800,
    0x0008_4808_0224_2088, 0x0888_0404_A00E_0880, 0x0460_2630_0508_0080, 0x7201_0200_8008_0080,
    0x0220_1084_0000_8021, 0x0801_8401_0002_9008, 0x3804_008A_0800_8800, 0xE024_1144_4082_0100,
    0xE088_0404_2130_0480, 0x4414_0301_1820_9042, 0x1000_2022_7002_6800, 0x0108_2042_0080_0800,
    0x6001_8861_0400_80C0, 0x7130_2110_0102_8220, 0x0010_1483_0040_2400, 0x0010_1404_8080_8031,
    0x0104_0402_2212_0404, 0x0040_4124_0120_8011, 0x0010_0426_0310_1080, 0x0000_0100_8404_0008,
    0x2100_0230_0212_0000, 0x2400_4009_2101_0000, 0x0060_4410_0A00_4140, 0x0109_0218_0200_2218,
    0x800C_8200_8084_4000, 0x0000_4021_0110_1008, 0x0004_20A2_0042_0820, 0x2048_0800_2420_9808,
    0x0100_0000_1002_0200, 0x8018_00E0_2042_0292, 0x8400_4048_1A04_0148, 0x8602_8810_0082_0040,
];

const fn rook_mask(square: Square) -> u64 {
    let (rank, file) = (square / 8, square % 8);

    let rank_mask = RANK_MASKS[rank as usize] & !(FILE_A | FILE_H);
    let file_mask = FILE_MASKS[file as usize] & !(RANK_MASKS[0] | RANK_MASKS[7]);

    (rank_mask | file_mask) & !(1u64 << square)
}

const fn bishop_mask(square: Square) -> u64 {
    let (rank, file) = ((square / 8) as i8, (square % 8) as i8);
    ray(rank, file, 1, 1) | ray(rank, file, 1, -1) | ray(rank, file, -1, 1) | ray(rank, file, -1, -1)
}

const fn ray(mut r: i8, mut f: i8, dr: i8, df: i8) -> u64 {
    let mut mask = 0;
    r += dr;
    f += df;
    while r > 0 && r < 7 && f > 0 && f < 7 {
        mask |= 1u64 << (r * 8 + f);
        r += dr;
        f += df;
    }
    mask
}

fn calc_atk_slider_slow(pt: PieceType, square: Square, occupancy: Bitboard) -> Bitboard {
    let (rank, file) = (square / 8, square % 8);

    let dirs: &[(i8, i8)] = match pt {
        PieceType::Rook => &[(1, 0), (-1, 0), (0, 1), (0, -1)],
        PieceType::Bishop => &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
    };

    let mut attacks = 0u64;
    for &(dr, df) in dirs {
        let (mut rf, mut cf) = ((rank as i8) + dr, (file as i8) + df);

        while (0..8).contains(&rf) && (0..8).contains(&cf) {
            let sq_bit = 1u64 << (rf * 8 + cf);
            attacks |= sq_bit;
            if (occupancy & sq_bit) != 0 {
                break;
            }
            rf += dr;
            cf += df;
        }
    }
    attacks
}

fn main() {
    println!("cargo:rerun-if-changed=src/core/bitboard.rs");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("magics.rs");
    let mut f = File::create(&dest_path).unwrap();

    let mut rooks = Vec::with_capacity(64);
    let mut bishops = Vec::with_capacity(64);
    let mut table = Vec::with_capacity(1_000_000);

    for sq in 0..64 {
        init_magic(sq, PieceType::Rook, &mut rooks, &mut table);
        init_magic(sq, PieceType::Bishop, &mut bishops, &mut table);
    }

    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let is_bmi2 = target_features.contains("bmi2");

    writeln!(f, "// Auto-generated by build.rs").unwrap();
    writeln!(f).unwrap();

    // Generate Rooks
    writeln!(f, "pub static ROOKS: [MagicEntry; 64] = [").unwrap();
    for entry in &rooks {
        if is_bmi2 {
            writeln!(f, "    MagicEntry {{ mask: {:#018X}, offset: {} }},", entry.mask, entry.offset).unwrap();
        } else {
            writeln!(
                f,
                "    MagicEntry {{ mask: {:#018X}, magic: {:#018X}, shift: {}, offset: {} }},",
                entry.mask, entry.magic, entry.shift, entry.offset
            )
            .unwrap();
        }
    }
    writeln!(f, "];").unwrap();

    // Generate Bishops
    writeln!(f, "pub static BISHOPS: [MagicEntry; 64] = [").unwrap();
    for entry in &bishops {
        if is_bmi2 {
            writeln!(f, "    MagicEntry {{ mask: {:#018X}, offset: {} }},", entry.mask, entry.offset).unwrap();
        } else {
            writeln!(
                f,
                "    MagicEntry {{ mask: {:#018X}, magic: {:#018X}, shift: {}, offset: {} }},",
                entry.mask, entry.magic, entry.shift, entry.offset
            )
            .unwrap();
        }
    }
    writeln!(f, "];").unwrap();

    // Generate Attack Table
    writeln!(f, "pub static ATTACK_TABLE: [u64; {}] = [", table.len()).unwrap();
    for chunk in table.chunks(8) {
        write!(f, "    ").unwrap();
        for &val in chunk {
            write!(f, "{val:#018X}, ").unwrap();
        }
        writeln!(f).unwrap();
    }
    writeln!(f, "];").unwrap();

    // Generate Lines
    let (lines, between) = init_lines_between();
    writeln!(f, "pub static LINES: [[u64; 64]; 64] = [").unwrap();
    for row in lines {
        write!(f, "    [").unwrap();
        for val in row {
            write!(f, "{val:#018X}, ").unwrap();
        }
        writeln!(f, "],").unwrap();
    }
    writeln!(f, "];").unwrap();

    // Generate Between
    writeln!(f, "pub static BETWEEN: [[u64; 64]; 64] = [").unwrap();
    for row in between {
        write!(f, "    [").unwrap();
        for val in row {
            write!(f, "{val:#018X}, ").unwrap();
        }
        writeln!(f, "],").unwrap();
    }
    writeln!(f, "];").unwrap();
}

fn init_lines_between() -> ([[u64; 64]; 64], [[u64; 64]; 64]) {
    let mut lines = [[0u64; 64]; 64];
    let mut between = [[0u64; 64]; 64];

    for s1 in 0..64u8 {
        for s2 in 0..64u8 {
            if s1 == s2 {
                continue;
            }

            let (r1, c1) = (s1 / 8, s1 % 8);
            let (r2, c2) = (s2 / 8, s2 % 8);

            let dr = i16::from(r2) - i16::from(r1);
            let dc = i16::from(c2) - i16::from(c1);

            let aligned = dr == 0 || dc == 0 || dr.abs() == dc.abs();

            let line = if aligned {
                let (step_r, step_c) = (dr.signum(), dc.signum());
                let mut bb = 0u64;
                for direction in [-1i16, 1] {
                    let (mut r, mut c) = (i16::from(r1), i16::from(c1));
                    while (0..8).contains(&r) && (0..8).contains(&c) {
                        bb |= 1u64 << (r * 8 + c);
                        r += direction * step_r;
                        c += direction * step_c;
                    }
                }
                bb
            } else {
                0
            };
            lines[usize::from(s1)][usize::from(s2)] = line;
        }
    }

    for s1 in 0..64u8 {
        for s2 in 0..64u8 {
            if lines[s1 as usize][s2 as usize] == 0 {
                continue;
            }

            let (r1, c1) = (s1 / 8, s1 % 8);
            let (r2, c2) = (s2 / 8, s2 % 8);

            let step_r = (i16::from(r2) - i16::from(r1)).signum();
            let step_c = (i16::from(c2) - i16::from(c1)).signum();

            let (mut r, mut c) = (i16::from(r1) + step_r, i16::from(c1) + step_c);
            let mut mask = 0u64;

            while r != i16::from(r2) || c != i16::from(c2) {
                mask |= 1u64 << (r * 8 + c);
                r += step_r;
                c += step_c;
            }
            between[s1 as usize][s2 as usize] = mask;
        }
    }
    (lines, between)
}

const fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn init_magic(square: Square, pt: PieceType, entries: &mut Vec<MagicEntry>, table: &mut Vec<u64>) {
    let mask = match pt {
        PieceType::Rook => rook_mask(square),
        PieceType::Bishop => bishop_mask(square),
    };

    let bits = mask.count_ones();
    let permutations = 1 << bits;
    let shift = 64 - u8::try_from(bits).unwrap();

    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let is_bmi2 = target_features.contains("bmi2");

    let mut occupancies = Vec::with_capacity(permutations as usize);
    let mut ref_attacks = Vec::with_capacity(permutations as usize);

    for i in 0..permutations {
        let mut occ = 0u64;
        let mut bits_processed = 0u32;
        let mut remaining_mask = mask;

        while remaining_mask != 0 {
            let bit_pos = remaining_mask.trailing_zeros();
            if (i & (1 << bits_processed)) != 0 {
                occ |= 1u64 << bit_pos;
            }
            remaining_mask &= remaining_mask - 1;
            bits_processed += 1;
        }
        occupancies.push(occ);
        ref_attacks.push(calc_atk_slider_slow(pt, square, occ));
    }

    let offset = u32::try_from(table.len()).unwrap();
    let mut magic = match pt {
        PieceType::Rook => ROOK_MAGICS[square as usize],
        PieceType::Bishop => BISHOP_MAGICS[square as usize],
    };

    if is_bmi2 {
        for attacks in ref_attacks {
            table.push(attacks);
        }
    } else {
        let mut seed = 0xDEAD_BEEF_CAFE_BABE ^ (u64::from(square) << 32);
        loop {
            let mut test_table = vec![0u64; permutations as usize];
            let mut collision = false;
            for i in 0..permutations as usize {
                let idx = (occupancies[i].wrapping_mul(magic) >> shift) as usize;
                if test_table[idx] != 0 && test_table[idx] != ref_attacks[i] {
                    collision = true;
                    break;
                }
                test_table[idx] = ref_attacks[i];
            }

            if !collision {
                for val in test_table {
                    table.push(val);
                }
                break;
            }
            magic = xorshift64(&mut seed) & xorshift64(&mut seed) & xorshift64(&mut seed);
        }
    }

    entries.push(MagicEntry { mask, magic, shift, offset });
}
