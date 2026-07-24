# WorldAtlas Arena — Sign In (OAuth)

The site uses a small Node.js auth layer (`server-auth.cjs`) bundled with the static SPA server (`server.cjs`). OAuth secrets stay on the server; the browser only receives a signed session cookie after a successful provider callback.

Supported providers:

- GitHub
- Google
- Hugging Face

## Quick start (local)

1. Copy the example env file:

   ```bash
   cp site/.env.example site/.env
   ```

2. Register OAuth apps for each provider you want (see below) and paste client IDs/secrets into `site/.env`.

3. Restart the site server:

   ```bash
   cd site
   ./start-worldarena-site.sh
   ```

4. Open `http://127.0.0.1:4173/arena/`, click **Sign In**, and choose a provider.

When no OAuth provider is configured, the sign-in modal shows disabled provider buttons and setup instructions. Guest sign-in is **off by default**; set `WORLDARENA_ALLOW_GUEST_AUTH=true` only for local dev without OAuth.

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `WORLDARENA_SITE_URL` | Recommended | Public site URL, no trailing slash. Used to build OAuth redirect URIs. Default: `http://127.0.0.1:4173` |
| `WORLDARENA_AUTH_SECRET` | Production | HMAC secret for session cookies. Change from the dev default before deploying. |
| `WORLDARENA_GITHUB_CLIENT_ID` | For GitHub | OAuth app client ID |
| `WORLDARENA_GITHUB_CLIENT_SECRET` | For GitHub | OAuth app client secret |
| `WORLDARENA_GOOGLE_CLIENT_ID` | For Google | OAuth client ID |
| `WORLDARENA_GOOGLE_CLIENT_SECRET` | For Google | OAuth client secret |
| `WORLDARENA_HF_CLIENT_ID` | For Hugging Face | OAuth app client ID |
| `WORLDARENA_HF_CLIENT_SECRET` | For Hugging Face | OAuth app client secret |
| `WORLDARENA_ALLOW_GUEST_AUTH` | Optional | `true` enables dev-only guest sign-in when no OAuth provider is configured. Default: **off**. |
| `WORLDARENA_REQUIRE_AUTH_FOR_VOTES` | Optional | `false` allows anonymous arena votes. Default: `true`. |

## OAuth app registration

Replace `SITE_URL` with your `WORLDARENA_SITE_URL` value.

### GitHub

1. Go to [GitHub Developer Settings → OAuth Apps](https://github.com/settings/developers).
2. Create a new OAuth App.
3. **Authorization callback URL:** `SITE_URL/api/auth/oauth/github/callback`
4. Copy Client ID and generate a Client Secret into `WORLDARENA_GITHUB_CLIENT_*`.

Scopes used: `read:user`, `user:email`.

### Google

1. Go to [Google Cloud Console → Credentials](https://console.cloud.google.com/apis/credentials).
2. Create an **OAuth 2.0 Client ID** (Web application).
3. **Authorized redirect URI:** `SITE_URL/api/auth/oauth/google/callback`
4. Copy Client ID and Client Secret into `WORLDARENA_GOOGLE_CLIENT_*`.

Scopes used: `openid`, `email`, `profile`.

### Hugging Face

1. Go to [Hugging Face → Settings → Applications](https://huggingface.co/settings/applications).
2. Create an OAuth application.
3. **Redirect URL:** `SITE_URL/api/auth/oauth/huggingface/callback`
4. Copy Client ID and Client Secret into `WORLDARENA_HF_CLIENT_*`.

Scopes used: `openid`, `profile`.

## Auth API (same origin)

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/auth/config` | GET | Public auth configuration and enabled providers |
| `/api/auth/session` | GET | Current user + config |
| `/api/auth/logout` | POST | Clear session cookie |
| `/api/auth/guest` | POST | Dev-only guest sign-in (disabled when OAuth is configured) |
| `/api/auth/oauth/{github\|google\|huggingface}` | GET | Start OAuth redirect |
| `/api/auth/oauth/{provider}/callback` | GET | OAuth callback, sets session cookie |

## Verify configuration

```bash
curl -s http://127.0.0.1:4173/api/auth/config | jq .
```

Each provider in `providers` should show `"enabled": true` after credentials are loaded and the server restarted.
