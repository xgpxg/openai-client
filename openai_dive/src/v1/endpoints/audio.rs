use crate::v1::api::Client;
use crate::v1::error::APIError;
use crate::v1::resources::audio::AudioSpeechParameters;
use crate::v1::resources::audio::AudioSpeechResponse;
#[cfg(feature = "stream")]
use crate::v1::resources::audio::AudioSpeechResponseChunkResponse;
use crate::v1::resources::audio::{AudioTranscriptionParameters, AudioTranslationParameters};
#[cfg(feature = "stream")]
use futures::Stream;
#[cfg(feature = "stream")]
use futures::StreamExt;
use serde_json::Value;
#[cfg(feature = "stream")]
use std::pin::Pin;

pub struct Audio<'a> {
    pub client: &'a Client,
    /// A converter for request parameters, needed because some providers have different parameter formats.
    /// We need to convert OpenAI format -> Non OpenAI format
    pub request_converter: Option<Box<dyn Fn(&Value) -> Value + Send + Sync>>,
}

impl Client {
    /// Learn how to turn audio into text or text into audio.
    pub fn audio(&self) -> Audio<'_> {
        Audio {
            client: self,
            request_converter: None,
        }
    }
}

impl Audio<'_> {
    /// Set a converter for request parameters.
    /// Before sending the request, we convert the parameters to the format needed by the provider.
    pub fn set_request_converter(
        &mut self,
        converter: Box<dyn Fn(&Value) -> Value + Send + Sync>,
    ) -> &mut Self {
        self.request_converter = Some(converter);
        self
    }
    /// Generates audio from the input text.
    pub async fn create_speech(
        &self,
        parameters: AudioSpeechParameters,
    ) -> Result<AudioSpeechResponse, APIError> {
        let bytes = match &self.request_converter {
            Some(converter) => {
                let params_value = serde_json::to_value(&parameters)
                    .map_err(|e| APIError::ParseError(e.to_string()))?;
                let converted_params = converter(&params_value);
                self.client
                    .post_raw("/audio/speech", &converted_params)
                    .await?
            }
            None => self.client.post_raw("/audio/speech", &parameters).await?,
        };
        // let bytes = self.client.post_raw("/audio/speech", &parameters).await?;

        Ok(AudioSpeechResponse { bytes })
    }

    /// Transcribes audio into the input language.
    pub async fn create_transcription(
        &self,
        parameters: AudioTranscriptionParameters,
    ) -> Result<String, APIError> {
        let mut form = reqwest::multipart::Form::new();

        let file = parameters.file.into_part().await?;

        form = form.part("file", file);

        form = form.text("model", parameters.model);

        if let Some(prompt) = parameters.prompt {
            form = form.text("prompt", prompt);
        }

        if let Some(language) = parameters.language {
            form = form.text("language", language.to_string());
        }

        if let Some(chunking_strategy) = parameters.chunking_strategy {
            form = form.text("chunking_strategy", chunking_strategy.to_string());
        }

        if let Some(response_format) = parameters.response_format {
            form = form.text("response_format", response_format.to_string());
        }

        if let Some(stream) = parameters.stream {
            form = form.text("stream", stream.to_string());
        }

        if let Some(temperature) = parameters.temperature {
            form = form.text("temperature", temperature.to_string());
        }

        if let Some(timestamp_granularities) = parameters.timestamp_granularities {
            form = form.text(
                "timestamp_granularities",
                timestamp_granularities
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>()
                    .join(","),
            );
        }

        if let Some(extra_body) = parameters.extra_body {
            match extra_body {
                Value::Object(map) => {
                    for (key, value) in map {
                        form = form.text(key, value.to_string());
                    }
                }
                _ => {
                    return Err(APIError::BadRequestError(
                        "extra_body must be formatted as a map of key: value".to_string(),
                    ));
                }
            }
        }

        let response = self
            .client
            .post_with_form("/audio/transcriptions", form)
            .await?;

        Ok(response)
    }

    /// Translates audio into English.
    pub async fn create_translation(
        &self,
        parameters: AudioTranslationParameters,
    ) -> Result<String, APIError> {
        let mut form = reqwest::multipart::Form::new();

        let file = parameters.file.into_part().await?;
        form = form.part("file", file);

        form = form.text("model", parameters.model);

        if let Some(prompt) = parameters.prompt {
            form = form.text("prompt", prompt);
        }

        if let Some(response_format) = parameters.response_format {
            form = form.text("response_format", response_format.to_string());
        }

        if let Some(temperature) = parameters.temperature {
            form = form.text("temperature", temperature.to_string());
        }

        let response = self
            .client
            .post_with_form("/audio/translations", form)
            .await?;

        Ok(response)
    }

    #[cfg(feature = "stream")]
    /// Generates audio from the input text.
    pub async fn create_speech_stream(
        &self,
        parameters: AudioSpeechParameters,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<AudioSpeechResponseChunkResponse, APIError>> + Send>>,
        APIError,
    > {
        use crate::v1::resources::audio::StreamAudioSpeechParameters;

        let stream_parameters = StreamAudioSpeechParameters {
            model: parameters.model,
            input: parameters.input,
            voice: parameters.voice,
            voice_text: None,
            response_format: parameters.response_format,
            speed: parameters.speed,
            stream: true,
        };

        let stream = Box::pin(
            self.client
                .post_stream_raw("/audio/speech", &stream_parameters)
                .await
                .unwrap()
                .map(|item| item.map(|bytes| AudioSpeechResponseChunkResponse { bytes })),
        );

        Ok(stream)
    }
}
