//! Print the search param table, or overwrite defaults with a finished tune's
//! values.

use std::{fs, io, io::IsTerminal, process};

use crate::engine::search_params::{ParamDef, frozen_param_defs, spsa_table, tunable_param_defs};

const PARAMS_FILE: &str = "src/engine/search_params.rs";
const TABLE_FILE: &str = "spsa.txt";
const SCREENFUL: usize = 70;

struct Change {
    param: &'static ParamDef,
    new: i32,
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
    let params = table.lines().count();
    if params > SCREENFUL && io::stdout().is_terminal() {
        match fs::write(TABLE_FILE, &table) {
            Ok(()) => println!("{params} params written to {TABLE_FILE}"),
            Err(e) => fail(&format!("{TABLE_FILE}: {e}")),
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

    for def in frozen_param_defs() {
        if find_value(&text, def.name).is_some() {
            eprintln!("spsa: {} is frozen; its reported value is ignored", def.name);
        }
    }

    if changes.is_empty() {
        fail(&format!("{report} contains no tunable changes"));
    }

    if let Err(message) = validate_changes(&changes) {
        fail(&message);
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
        .filter_map(|def| {
            let new = find_value(text, def.name)?;
            (new != def.default).then_some(Change { param: def, new })
        })
        .collect()
}

fn validate_changes(changes: &[Change]) -> Result<(), String> {
    for change in changes {
        let param = change.param;
        if !(param.min..=param.max).contains(&change.new) {
            return Err(format!("{} = {} is outside {}..={}", param.name, change.new, param.min, param.max));
        }
    }
    Ok(())
}

fn find_value(text: &str, name: &str) -> Option<i32> {
    for (at, _) in text.match_indices(name) {
        let before = text[..at].chars().next_back();
        let tail = &text[at + name.len()..];

        if before.is_some_and(is_name_char) || tail.starts_with(is_name_char) {
            continue;
        }

        if let Some(value) = tail
            .split(|c: char| c.is_whitespace() || matches!(c, ',' | ':' | '=' | '"' | '(' | ')' | '[' | ']' | '{' | '}'))
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
    let width = changes.iter().map(|c| c.param.name.len()).max().unwrap_or(0);

    for change in changes {
        let param = change.param;
        let delta = change.new - param.default;
        // A bound value means the range limited the tune, not that the value is right.
        let pinned = if change.new == param.min || change.new == param.max { "  (at bound)" } else { "" };
        println!("{:<width$}  {:>7} -> {:<7} {delta:+}{pinned}", param.name, param.default, change.new);
    }
}

fn rewrite(source: &str, changes: &[Change]) -> (String, usize) {
    let mut out = String::with_capacity(source.len());
    let mut applied = 0;

    for line in source.split_inclusive('\n') {
        match changes.iter().find(|c| declares(line, c.param.name)) {
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
    fn a_value_is_read_out_of_common_report_formats() {
        for report in [
            "lmr_base, int, 117, 10, 160, 7, 0.002",
            "lmr_base=117",
            "\"lmr_base\": 117,",
            "{\"lmr_base\": 117}",
            "lmr_base 117",
            "param_lmr_base 9\nlmr_base 117",
        ] {
            assert_eq!(find_value(report, "lmr_base"), Some(117), "{report}");
        }
        assert_eq!(find_value("lmr_base_extra 117", "lmr_base"), None);
    }

    #[test]
    fn an_out_of_range_value_is_rejected() {
        let param = tunable_param_defs().find(|p| p.name == "lmr_base").unwrap();
        assert!(validate_changes(&[Change { param, new: param.min }]).is_ok());
        assert!(validate_changes(&[Change { param, new: param.max }]).is_ok());
        assert!(validate_changes(&[Change { param, new: param.max + 1 }]).is_err());
    }

    #[test]
    fn a_rewrite_holds_the_column() {
        let source = "        T (lmr_base,          100,  10),\n        T (lmr_divisor,       225,   1,  350),\n";
        let change = |name, new| Change { param: tunable_param_defs().find(|p| p.name == name).unwrap(), new };
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
