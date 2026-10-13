//! Package-bound text tokenization and a single chat-template rendering boundary.

use std::io::{self, Write};

use anyhow::{Context, Result, anyhow, ensure};
use minijinja::{Environment, Error, ErrorKind, context};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::package::Package;

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    #[default]
    Medium,
    Xhigh,
}

pub struct ChatOptions {
    pub thinking: bool,
    pub preserve_thinking: bool,
    pub effort: ReasoningEffort,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            thinking: true,
            preserve_thinking: true,
            effort: ReasoningEffort::Medium,
        }
    }
}

pub struct RenderedPrompt {
    pub text: String,
    pub tokens: Vec<u32>,
}

pub struct TextTokenizer {
    tokenizer: Tokenizer,
    environment: Environment<'static>,
    stop_ids: Vec<u32>,
    package_manifest_digest: [u8; 32],
    assets_digest: [u8; 32],
}

fn hash_asset(hasher: &mut Sha256, name: &str, bytes: &[u8]) -> Result<()> {
    hasher.update(u64::try_from(name.len())?.to_le_bytes());
    hasher.update(name.as_bytes());
    hasher.update(u64::try_from(bytes.len())?.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

// Transformers' template filter uses JSON separators with spaces, preserves
// insertion order, and emits Unicode and HTML characters without escaping.
struct TemplateJson;

fn write_python_float<W: ?Sized + Write>(writer: &mut W, value: f64) -> io::Result<()> {
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "nonfinite template number"))?;
    let exponent: i32 = exponent
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid template exponent"))?;
    if !(-4..16).contains(&exponent) {
        return write!(writer, "{mantissa}e{exponent:+03}");
    }
    let negative = mantissa.starts_with('-');
    let digits = mantissa.trim_start_matches('-').replace('.', "");
    if negative {
        writer.write_all(b"-")?;
    }
    let point = exponent + 1;
    if point <= 0 {
        write!(writer, "0.{}{}", "0".repeat((-point) as usize), digits)
    } else if point as usize >= digits.len() {
        write!(
            writer,
            "{}{}.0",
            digits,
            "0".repeat(point as usize - digits.len())
        )
    } else {
        write!(
            writer,
            "{}.{}",
            &digits[..point as usize],
            &digits[point as usize..]
        )
    }
}

impl serde_json::ser::Formatter for TemplateJson {
    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        write_python_float(writer, value)
    }

    fn write_f32<W: ?Sized + Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        write_python_float(writer, f64::from(value))
    }
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

fn template_json(value: minijinja::Value) -> std::result::Result<String, Error> {
    let mut bytes = Vec::new();
    value
        .serialize(&mut serde_json::Serializer::with_formatter(
            &mut bytes,
            TemplateJson,
        ))
        .map_err(|err| Error::new(ErrorKind::InvalidOperation, err.to_string()))?;
    String::from_utf8(bytes).map_err(|err| Error::new(ErrorKind::InvalidOperation, err.to_string()))
}

impl TextTokenizer {
    pub fn load(package: &Package) -> Result<Self> {
        let tokenizer_bytes = std::fs::read(package.tokenizer_file("tokenizer.json")?)?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|err| anyhow!("load package tokenizer: {err}"))?;
        let generation_bytes = std::fs::read(package.tokenizer_file("generation_config.json")?)?;
        let generation: Value = serde_json::from_slice(&generation_bytes)?;
        let declared = &generation["eos_token_id"];
        let values = declared
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_else(|| std::slice::from_ref(declared));
        let stop_ids: Vec<u32> = values
            .iter()
            .map(|value| {
                Ok(u32::try_from(
                    value.as_u64().context("invalid generation stop token")?,
                )?)
            })
            .collect::<Result<_>>()?;
        ensure!(
            !stop_ids.is_empty()
                && stop_ids
                    .iter()
                    .all(|&id| (id as usize) < tokenizer.get_vocab_size(true)),
            "invalid generation stop token inventory"
        );
        let template_bytes = std::fs::read(package.tokenizer_file("chat_template.jinja")?)?;
        let mut assets = Sha256::new();
        hash_asset(&mut assets, "tokenizer.json", &tokenizer_bytes)?;
        hash_asset(&mut assets, "generation_config.json", &generation_bytes)?;
        hash_asset(&mut assets, "chat_template.jinja", &template_bytes)?;
        let template = String::from_utf8(template_bytes).context("chat template is not UTF-8")?;
        let mut environment = Environment::new();
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_filter("tojson", template_json);
        environment.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        environment.add_template_owned("chat", template)?;
        Ok(Self {
            tokenizer,
            environment,
            stop_ids,
            package_manifest_digest: package.manifest_digest(),
            assets_digest: assets.finalize().into(),
        })
    }

    pub(crate) fn checkpoint_identity(&self) -> ([u8; 32], [u8; 32]) {
        (self.package_manifest_digest, self.assets_digest)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|err| anyhow!("encode prompt: {err}"))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|err| anyhow!("decode tokens: {err}"))
    }

    pub fn stop_ids(&self) -> &[u32] {
        &self.stop_ids
    }

    /// Decode incrementally while retaining incomplete UTF-8 bytes between tokens.
    pub fn stream_decoder(&self) -> impl FnMut(u32) -> Result<Option<String>> + '_ {
        let mut stream = self.tokenizer.decode_stream(false);
        move |token| {
            stream
                .step(token)
                .map_err(|err| anyhow!("decode streamed token: {err}"))
        }
    }

    pub fn render(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        options: &ChatOptions,
    ) -> Result<RenderedPrompt> {
        ensure!(!messages.is_empty(), "chat requires at least one message");
        for message in messages {
            if let Some(parts) = message.get("content").and_then(Value::as_array) {
                ensure!(
                    parts
                        .iter()
                        .all(|part| part["type"] == "text" && part["text"].is_string()),
                    "only text content is supported"
                );
            }
        }
        let text = self
            .environment
            .get_template("chat")?
            .render(context! {
                messages => messages,
                tools => tools.unwrap_or(&[]),
                enable_thinking => options.thinking,
                preserve_thinking => options.preserve_thinking,
                reasoning_effort => options.effort,
                add_generation_prompt => true,
            })
            .context("render package chat template")?;
        let tokens = self.encode(&text)?;
        Ok(RenderedPrompt { text, tokens })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendered_tokens_and_streamed_utf8_preserve_the_prompt() -> Result<()> {
        use tokenizers::{models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

        let vocabulary = [
            ("u", 0),
            ("s", 1),
            ("e", 2),
            ("r", 3),
            (":", 4),
            ("Ġ", 5),
            ("c", 6),
            ("a", 7),
            ("f", 8),
            ("Ã", 9),
            ("©", 10),
            ("!", 11),
        ]
        .map(|(token, id)| (token.to_owned(), id));
        let model = BPE::builder()
            .vocab_and_merges(vocabulary, vec![])
            .build()
            .map_err(|error| anyhow!("synthetic tokenizer: {error}"))?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        let mut environment = Environment::new();
        environment.add_template(
            "chat",
            concat!(
                "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}",
                "{% if add_generation_prompt and enable_thinking and preserve_thinking ",
                "and reasoning_effort == 'xhigh' and tools|length == 1 %}!{% endif %}",
            ),
        )?;
        let tokenizer = TextTokenizer {
            tokenizer,
            environment,
            stop_ids: vec![11],
            package_manifest_digest: [0; 32],
            assets_digest: [0; 32],
        };
        let messages = [serde_json::json!({"role": "user", "content": "café"})];
        let tools = [serde_json::json!({"name": "lookup"})];
        let options = ChatOptions {
            thinking: true,
            preserve_thinking: true,
            effort: ReasoningEffort::Xhigh,
        };
        let prompt = tokenizer.render(&messages, Some(&tools), &options)?;
        assert_eq!(prompt.text, "user: café!");
        assert_eq!(prompt.tokens, (0..12).collect::<Vec<_>>());
        assert_eq!(tokenizer.decode(&prompt.tokens, false)?, prompt.text);
        assert_eq!(tokenizer.stop_ids(), &[11]);
        let mut decoder = tokenizer.stream_decoder();
        let mut streamed = String::new();
        for &id in &prompt.tokens {
            let piece = decoder(id)?;
            if id == 9 {
                assert!(piece.is_none(), "incomplete UTF-8 must remain buffered");
            }
            if let Some(piece) = piece {
                streamed.push_str(&piece);
            }
        }
        assert_eq!(streamed, prompt.text);
        assert!(tokenizer.render(&[], None, &options).is_err());
        let image = [serde_json::json!({"role": "user", "content": [{"type": "image"}]})];
        assert!(tokenizer.render(&image, None, &options).is_err());
        Ok(())
    }

    #[test]
    fn template_json_preserves_order_spacing_and_unicode() {
        let value: Value = serde_json::from_str(r#"{"z":"<caffè>","a":[1,true,null]}"#).unwrap();
        let text = template_json(minijinja::Value::from_serialize(value)).unwrap();
        assert_eq!(text, r#"{"z": "<caffè>", "a": [1, true, null]}"#);
    }

    #[test]
    fn template_float_spelling_matches_python_json() {
        for (value, expected) in [
            (1e-7, "1e-07"),
            (1e20, "1e+20"),
            (-0.0, "-0.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.234e-4, "0.0001234"),
            (0.125, "0.125"),
            (1.5e-5, "1.5e-05"),
        ] {
            let mut bytes = Vec::new();
            write_python_float(&mut bytes, value).unwrap();
            assert_eq!(String::from_utf8(bytes).unwrap(), expected);
        }
    }
}
