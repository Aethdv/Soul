//! Terminal formatting utilities.

use std::io::{IsTerminal, stdout};

use crate::color::{ASH_PEN, BOLD, CORAL_PEN, HAZE_PEN, IVORY_PEN, LILAC_PEN, RESET, STEEL_PEN};

/// Section titles.
pub const HEADER: &str = LILAC_PEN;
/// Argument placeholders.
pub const ARG: &str = CORAL_PEN;
/// Command and flag names: neutral and bright, since this is the text you type.
pub const CMD: &str = IVORY_PEN;
/// Descriptions, cooled so the names carry the scan.
pub const DESC: &str = HAZE_PEN;
/// The dash between a name and its description.
pub const RULE: &str = ASH_PEN;
/// A default's value inside its parenthesis.
pub const VALUE: &str = STEEL_PEN;

pub fn banner() {
    if stdout().is_terminal() {
        println!("{BOLD}{LILAC_PEN}✦ Soul v{}{RESET}", crate::VERSION);
    }
}

pub struct Help {
    width: usize,
    use_ansi: bool,
}

impl Help {
    /// Colors when a terminal is reading, plain text when the output is redirected.
    pub fn new(width: usize) -> Self { Self { width, use_ansi: stdout().is_terminal() } }

    /// Overrides the terminal check, for a pager that renders escapes anyway.
    pub fn with_ansi(mut self, enabled: bool) -> Self {
        self.use_ansi = enabled;
        self
    }

    pub fn header(&self, title: &str) {
        if self.use_ansi {
            println!("{BOLD}{HEADER}{title}{RESET}");
        } else {
            println!("{title}");
        }
    }

    pub fn separator(&self) {
        println!();
    }

    pub fn command(&self, name: &str, desc: &str) { self.print_line_internal(2, name, "", desc); }

    pub fn command_args(&self, name: &str, args: &str, desc: &str) { self.print_line_internal(2, name, args, desc); }

    pub fn command_default(&self, name: &str, desc: &str, default: &str) {
        self.print_line_internal(2, name, "", &self.with_default(desc, default));
    }

    pub fn subcommand(&self, name: &str, args: &str, desc: &str) { self.print_line_internal(4, name, args, desc); }

    pub fn subcommand_default(&self, name: &str, args: &str, desc: &str, default: &str) {
        self.print_line_internal(4, name, args, &self.with_default(desc, default));
    }

    pub fn option(&self, flags: &str, value: &str, desc: &str) { self.print_line_internal(2, flags, value, desc); }

    pub fn option_default(&self, flags: &str, value: &str, desc: &str, default: &str) {
        self.print_line_internal(2, flags, value, &self.with_default(desc, default));
    }

    pub fn example(&self, cmd: &str) {
        println!("  {cmd}");
    }

    fn with_default(&self, desc: &str, default: &str) -> String {
        if self.use_ansi {
            format!("{desc} {ASH_PEN}(default: {VALUE}{default}{ASH_PEN})")
        } else {
            format!("{desc} (default: {default})")
        }
    }

    fn print_line_internal(&self, indent: usize, name: &str, args: &str, desc: &str) {
        let name_len = name.len();
        let args_len = args.len();
        let gap = usize::from(args_len > 0);
        let content_len = indent + name_len + gap + args_len;
        let padding = if content_len < self.width { self.width - content_len } else { 1 };
        let spaces_indent = " ".repeat(indent);
        let spaces_pad = " ".repeat(padding);

        if !self.use_ansi {
            if args.is_empty() {
                println!("{spaces_indent}{name}{spaces_pad}- {desc}");
            } else {
                println!("{spaces_indent}{name} {args}{spaces_pad}- {desc}");
            }
            return;
        }

        if args.is_empty() {
            println!("{spaces_indent}{CMD}{name}{spaces_pad}{RULE}- {DESC}{desc}{RESET}");
        } else {
            println!("{spaces_indent}{CMD}{name} {ARG}{args}{spaces_pad}{RULE}- {DESC}{desc}{RESET}");
        }
    }
}
