//! Format detection from a hint or by sniffing the content.

/// How content is parsed into structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    Markdown,
    Code,
    Html,
    Text,
}

impl Format {
    pub fn as_str(self) -> &'static str {
        match self {
            Format::Markdown => "markdown",
            Format::Code => "code",
            Format::Html => "html",
            Format::Text => "text",
        }
    }

    /// Parses a hint: a format name, MIME type, file name or extension.
    pub fn from_hint(hint: &str) -> Option<Format> {
        let h = hint.trim().to_ascii_lowercase();
        let ext = h.rsplit(['.', '/']).next().unwrap_or(&h);
        Some(match ext {
            "markdown" | "md" | "mdx" | "x-markdown" => Format::Markdown,
            "html" | "htm" | "xhtml" => Format::Html,
            "text" | "txt" | "plain" => Format::Text,
            "code" | "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "java" | "c" | "h"
            | "cc" | "cpp" | "hpp" | "cs" | "rb" | "php" | "swift" | "kt" | "scala" | "sh"
            | "lua" | "zig" | "sql" => Format::Code,
            _ => return None,
        })
    }
}

/// Picks a format: the hint if it names one, otherwise by sniffing.
pub fn detect(text: &str, hint: Option<&str>) -> Format {
    if let Some(f) = hint.and_then(Format::from_hint) {
        return f;
    }
    let head = &text[..floor_char_boundary(text, 8192)];
    let lower = head.to_ascii_lowercase();
    if lower.trim_start().starts_with("<!doctype html")
        || lower.contains("<html")
        || ["<p>", "<div", "<h1", "<h2", "<body"]
            .iter()
            .filter(|t| lower.contains(*t))
            .count()
            >= 2
    {
        return Format::Html;
    }

    let lines: Vec<&str> = head.lines().collect();
    let md = lines
        .iter()
        .filter(|l| {
            let t = l.trim_start();
            (t.starts_with('#') && t.trim_start_matches('#').starts_with(' '))
                || t.starts_with("```")
                || t.starts_with("- ")
                || t.starts_with("* ")
                || t.starts_with("> ")
        })
        .count();
    let code = lines
        .iter()
        .filter(|l| {
            let t = l.trim();
            t.ends_with('{')
                || t.ends_with(';')
                || t == "}"
                || [
                    "fn ",
                    "def ",
                    "class ",
                    "import ",
                    "#include",
                    "func ",
                    "function ",
                    "const ",
                    "let ",
                ]
                .iter()
                .any(|k| t.starts_with(k))
        })
        .count();
    let n = lines.len().max(1);
    if code * 4 >= n && code > md {
        Format::Code
    } else if md > 0 {
        Format::Markdown
    } else {
        Format::Text
    }
}

pub(crate) fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}
