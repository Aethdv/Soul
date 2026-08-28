//! The `spsa` subcommand: print the tuning table, or fold a finished tune's values
//! back into the declared defaults.

use std::{fs, io, io::IsTerminal, process};

use crate::engine::search_params::{frozen_param_defs, spsa_table, tunable_param_defs};

const PARAMS_FILE: &str = "src/engine/search_params.rs";
const SCREENFUL: usize = 70;

struct Change {
    name: &'static str,
    old: i32,
    new: i32,
    min: i32,
    max: i32,
}

pub fn run(args: &[&str]) {
    match args {
        [] => print_table(),
        ["apply", report, rest @ ..] => apply(report, rest),
        _ => fail("usage: soul spsa [apply <report> [--dry-run] [--params <path>]]"),
    }
}

fn print_table() {
    let table = spsa_table();
    if table.lines().count() > SCREENFUL && io::stdout().is_terminal() {
        match fs::write("spsa.txt", &table) {
            Ok(()) => println!("{} params written to spsa.txt", table.lines().count()),
            Err(e) => fail(&format!("spsa.txt: {e}")),
        }
    } else {
        print!("{table}");
    }
}

fn apply(report: &str, flags: &[&str]) {
    let mut dry_run = false;
    let mut params_file = PARAMS_FILE;
    let mut rest = flags.iter();

    while let Some(flag) = rest.next() {
        match *flag {
            "--dry-run" => dry_run = true,
            "--params" => match rest.next() {
                Some(path) => params_file = path,
                None => fail("--params wants a path"),
            },
            other => fail(&format!("unknown flag {other}")),
        }
    }

    let text = fs::read_to_string(report).unwrap_or_else(|e| fail(&format!("{report}: {e}")));
    let changes = read_report(&text);

    // A name frozen since the tune ran would otherwise be dropped without a word.
    for def in frozen_param_defs() {
        if find_value(&text, def.name).is_some() {
            eprintln!("spsa: {} is frozen; its reported value is ignored", def.name);
        }
    }

    if changes.is_empty() {
        fail(&format!("{report} names no tunable parameter"));
    }

    report_changes(&changes);

    if dry_run {
        return;
    }

    let source = fs::read_to_string(params_file)
        .unwrap_or_else(|e| fail(&format!("{params_file}: {e}. Run from the repository root, or pass --params")));

    let (rewritten, applied) = rewrite(&source, &changes);
    if applied != changes.len() {
        fail(&format!("{applied} of {} declarations found in {params_file}; nothing written", changes.len()));
    }

    match fs::write(params_file, rewritten) {
        Ok(()) => println!("\n{applied} defaults written to {params_file}"),
        Err(e) => fail(&format!("{params_file}: {e}")),
    }
}

fn read_report(text: &str) -> Vec<Change> {
    tunable_param_defs()
        .iter()
        .filter_map(|def| {
            let new = find_value(text, def.name)?;
            let old = def.default as i32;
            (new != old).then_some(Change { name: def.name, old, new, min: def.min as i32, max: def.max as i32 })
        })
        .collect()
}

/// The first integer following a whole-word `name`, so the `int` of a table row is skipped.
fn find_value(text: &str, name: &str) -> Option<i32> {
    for (at, _) in text.match_indices(name) {
        let before = text[..at].chars().next_back();
        let tail = &text[at + name.len()..];
        if before.is_some_and(is_name_char) || tail.starts_with(is_name_char) {
            continue;
        }

        if let Some(value) = tail
            .split(|c: char| c.is_whitespace() || matches!(c, ',' | ':' | '=' | '"' | '(' | ')'))
            .filter(|token| !token.is_empty())
            .find_map(|token| token.parse::<i32>().ok())
        {
            return Some(value);
        }
    }
    None
}

fn is_name_char(c: char) -> bool { c.is_alphanumeric() || c == '_' }

fn report_changes(changes: &[Change]) {
    let width = changes.iter().map(|c| c.name.len()).max().unwrap_or(0);

    for change in changes {
        let delta = change.new - change.old;
        // Pinned to an edge means the range fenced the tune in, not that this is the value.
        let pinned = if change.new == change.min || change.new == change.max { "  (at bound)" } else { "" };
        println!("{:<width$}  {:>7} -> {:<7} {delta:+}{pinned}", change.name, change.old, change.new);
    }
}

/// Replaces each named default, holding the column after it still where padding allows.
fn rewrite(source: &str, changes: &[Change]) -> (String, usize) {
    let mut out = String::with_capacity(source.len());
    let mut applied = 0;

    for line in source.split_inclusive('\n') {
        match changes.iter().find(|c| declares(line, c.name)) {
            Some(change) => {
                out.push_str(&replace_default(line, change.new));
                applied += 1;
            },
            None => out.push_str(line),
        }
    }
    (out, applied)
}

fn declares(line: &str, name: &str) -> bool {
    let body = line.trim_start();
    let body = body.strip_prefix("T (").or_else(|| body.strip_prefix("NT("));
    body.and_then(|body| body.split_once(',')).is_some_and(|(field, _)| field.trim() == name)
}

fn replace_default(line: &str, value: i32) -> String {
    let Some(comma) = line.find(',') else {
        return line.to_string();
    };
    let after = &line[comma + 1..];
    let pad = after.len() - after.trim_start().len();
    let number = &after[pad..];
    let digits = number.find(|c: char| !(c.is_ascii_digit() || c == '-')).unwrap_or(number.len());

    let new = value.to_string();
    let old_width = pad + digits;
    let padding = " ".repeat(old_width.saturating_sub(new.len()).max(1));

    format!("{}{padding}{new}{}", &line[..=comma], &after[pad + digits..])
}

fn fail(message: &str) -> ! {
    eprintln!("spsa: {message}");
    process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_is_read_out_of_every_report_format() {
        for report in [
            "lmr_base, int, 117, 10, 160, 7, 0.002",
            "lmr_base=117",
            "\"lmr_base\": 117,",
            "lmr_base 117",
            "param_lmr_base 9\nlmr_base 117",
        ] {
            assert_eq!(find_value(report, "lmr_base"), Some(117), "{report}");
        }
        assert_eq!(find_value("lmr_base_extra 117", "lmr_base"), None);
    }

    #[test]
    fn a_rewrite_holds_the_column() {
        let source = "        T (lmr_base,          100,  10),\n        T (lmr_divisor,       225,   1,  350),\n";
        let change = |name, new| Change { name, old: 0, new, min: 1, max: 350 };
        let changes = vec![change("lmr_base", 7), change("lmr_divisor", 1234)];

        let (out, applied) = rewrite(source, &changes);
        assert_eq!(applied, 2);
        assert_eq!(out, "        T (lmr_base,            7,  10),\n        T (lmr_divisor,      1234,   1,  350),\n");
    }

    #[test]
    fn a_frozen_declaration_is_matched_too() {
        assert!(declares("        NT(vol_king,     0),", "vol_king"));
        assert!(!declares("        NT(vol_king,     0),", "vol_kin"));
        assert!(!declares("    pub vol_king: i32,", "vol_king"));
    }
}
