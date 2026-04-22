//! Terminal formatting utilities.

pub const HEADER: &str = "\x1b[92m";
pub const ARG: &str = "\x1b[91m";
pub const CMD: &str = "\x1b[33m";
pub const RESET: &str = "\x1b[0m";

pub struct Help {
    width: usize,
    use_ansi: bool,
}

impl Help {
    pub fn new(width: usize) -> Self {
        Self { width, use_ansi: true }
    }

    pub fn with_ansi(mut self, enabled: bool) -> Self {
        self.use_ansi = enabled;
        self
    }

    pub fn header(&self, title: &str) {
        if self.use_ansi {
            println!("{HEADER}{title}{RESET}");
        } else {
            println!("{title}");
        }
    }

    pub fn separator(&self) {
        println!();
    }

    pub fn command(&self, name: &str, desc: &str) {
        self.print_line_internal(2, name, "", desc);
    }

    pub fn command_args(&self, name: &str, args: &str, desc: &str) {
        self.print_line_internal(2, name, args, desc);
    }

    pub fn command_default(&self, name: &str, desc: &str, default: &str) {
        let full_desc = if self.use_ansi {
            format!("{desc} \x1b[90m(default: \x1b[36m{default}\x1b[90m)\x1b[0m")
        } else {
            format!("{desc} (default: {default})")
        };
        self.print_line_internal(2, name, "", &full_desc);
    }

    pub fn subcommand(&self, name: &str, args: &str, desc: &str) {
        self.print_line_internal(4, name, args, desc);
    }

    pub fn subcommand_default(&self, name: &str, args: &str, desc: &str, default: &str) {
        let full_desc = if self.use_ansi {
            format!("{desc} \x1b[90m(default: \x1b[36m{default}\x1b[90m)\x1b[0m")
        } else {
            format!("{desc} (default: {default})")
        };
        self.print_line_internal(4, name, args, &full_desc);
    }

    pub fn option(&self, flags: &str, value: &str, desc: &str) {
        self.print_line_internal(2, flags, value, desc);
    }

    pub fn option_default(&self, flags: &str, value: &str, desc: &str, default: &str) {
        let full_desc = if self.use_ansi {
            format!("{desc} \x1b[90m(default: \x1b[36m{default}\x1b[90m)\x1b[0m")
        } else {
            format!("{desc} (default: {default})")
        };
        self.print_line_internal(2, flags, value, &full_desc);
    }

    pub fn example(&self, cmd: &str) {
        println!("  {cmd}");
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
            println!("{spaces_indent}{CMD}{name}{spaces_pad}{RESET}- {desc}");
        } else {
            println!("{spaces_indent}{CMD}{name} {ARG}{args}{spaces_pad}{RESET}- {desc}");
        }
    }
}
