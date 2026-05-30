# 20260504-audio-v13

- Add a default enabled looping soundtrack control for the uncertain archive.
- Use Merry Christmas Mr. Lawrence source slots under /assets/audio/ with mp3/m4a/ogg fallback.
- Respect browser autoplay policy by retrying on first pointer, keyboard, or wheel interaction.
- Add persistent mute/unmute state, compact bottom-right music UI, and static audio MIME/cache handling.
- Return 404 for missing static assets instead of serving the SPA shell as audio fallback.
