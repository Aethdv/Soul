//! Terminal formatting utilities.

use std::io::{IsTerminal, stdout};

use crate::color::{ASH_PEN, BOLD, CORAL_PEN, HAZE_PEN, IVORY_PEN, LILAC_PEN, RESET, STEEL_PEN};

const HEADER: &str = LILAC_PEN;
const ARGUMENT: &str = CORAL_PEN;
const COMMAND: &str = IVORY_PEN;
const DESCRIPTION: &str = HAZE_PEN;
const DIM: &str = ASH_PEN;
const DEFAULT_VALUE: &str = STEEL_PEN;

pub fn banner() {
    if stdout().is_terminal() {
        println!("{BOLD}{LILAC_PEN}✦ Soul v{}{RESET}", crate::VERSION);
    }
}

pub struct Help {
    description_col: usize,
    use_ansi: bool,
}

impl Help {
    /// Enables ANSI when stdout is a terminal.
    pub fn new(description_col: usize) -> Self { Self { description_col, use_ansi: stdout().is_terminal() } }

    /// Overrides ANSI detection.
    pub(crate) fn with_ansi(mut self, enabled: bool) -> Self {
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

    pub fn command_args(&self, name: &str, args: &str, desc: &str) { self.print_entry(2, name, args, desc); }
    pub(crate) fn command(&self, name: &str, desc: &str) { self.print_entry(2, name, "", desc); }
    pub(crate) fn command_default(&self, name: &str, desc: &str, default: &str) {
        self.print_entry(2, name, "", &self.format_default(desc, default));
    }

    pub fn subcommand(&self, name: &str, args: &str, desc: &str) { self.print_entry(4, name, args, desc); }
    pub fn subcommand_default(&self, name: &str, args: &str, desc: &str, default: &str) {
        self.print_entry(4, name, args, &self.format_default(desc, default));
    }

    pub fn option(&self, flags: &str, value: &str, desc: &str) { self.print_entry(2, flags, value, desc); }
    pub fn option_default(&self, flags: &str, value: &str, desc: &str, default: &str) {
        self.print_entry(2, flags, value, &self.format_default(desc, default));
    }

    pub fn example(&self, cmd: &str) {
        println!("  {cmd}");
    }

    fn format_default(&self, desc: &str, default: &str) -> String {
        if self.use_ansi {
            format!("{desc} {DIM}(default: {DEFAULT_VALUE}{default}{DIM})")
        } else {
            format!("{desc} (default: {default})")
        }
    }

    fn print_entry(&self, indent: usize, name: &str, args: &str, desc: &str) {
        let gap = usize::from(!args.is_empty());
        let used = indent + name.len() + gap + args.len();
        let padding = self.description_col.saturating_sub(used).max(1);
        let indent = " ".repeat(indent);
        let pad = " ".repeat(padding);

        let (command, argument, dim, description, reset) =
            if self.use_ansi { (COMMAND, ARGUMENT, DIM, DESCRIPTION, RESET) } else { ("", "", "", "", "") };

        if args.is_empty() {
            println!("{indent}{command}{name}{pad}{dim}- {description}{desc}{reset}");
        } else {
            println!("{indent}{command}{name} {argument}{args}{pad}{dim}- {description}{desc}{reset}");
        }
    }
}
