const crypto = require("crypto");
const fs = require("fs");
const https = require("https");
const path = require("path");

const SESSION_COOKIE = "worldarena_session";
const OAUTH_STATE_COOKIE = "worldarena_oauth_state";
const SESSION_TTL_MS = 1000 * 60 * 60 * 24 * 30;

function createAuthModule(options) {
  const {
    root,
    port,
    sendJson,
    sendBadRequest,
    parseJsonRequest
  } = options;

  const siteUrl = (process.env.WORLDARENA_SITE_URL || `http://127.0.0.1:${port}`).replace(/\/$/, "");
  const authSecret = process.env.WORLDARENA_AUTH_SECRET || "worldarena-dev-secret-change-me";
  const allowGuestAuth = process.env.WORLDARENA_ALLOW_GUEST_AUTH === "true";
  const requireAuthForVotes = process.env.WORLDARENA_REQUIRE_AUTH_FOR_VOTES !== "false";
  const feedbackLogPath = process.env.WORLDARENA_FEEDBACK_LOG || path.join(root, ".cache", "feedback.json");

  const oauthProviders = {
    github: {
      id: "github",
      label: "GitHub",
      enabled: Boolean(process.env.WORLDARENA_GITHUB_CLIENT_ID && process.env.WORLDARENA_GITHUB_CLIENT_SECRET),
      clientId: process.env.WORLDARENA_GITHUB_CLIENT_ID || "",
      clientSecret: process.env.WORLDARENA_GITHUB_CLIENT_SECRET || "",
      authorizeUrl: "https://github.com/login/oauth/authorize",
      tokenUrl: "https://github.com/login/oauth/access_token",
      scope: "read:user user:email",
      userUrl: "https://api.github.com/user"
    },
    google: {
      id: "google",
      label: "Google",
      enabled: Boolean(process.env.WORLDARENA_GOOGLE_CLIENT_ID && process.env.WORLDARENA_GOOGLE_CLIENT_SECRET),
      clientId: process.env.WORLDARENA_GOOGLE_CLIENT_ID || "",
      clientSecret: process.env.WORLDARENA_GOOGLE_CLIENT_SECRET || "",
      authorizeUrl: "https://accounts.google.com/o/oauth2/v2/auth",
      tokenUrl: "https://oauth2.googleapis.com/token",
      scope: "openid email profile",
      userUrl: "https://openidconnect.googleapis.com/v1/userinfo"
    },
    huggingface: {
      id: "huggingface",
      label: "Hugging Face",
      enabled: Boolean(process.env.WORLDARENA_HF_CLIENT_ID && process.env.WORLDARENA_HF_CLIENT_SECRET),
      clientId: process.env.WORLDARENA_HF_CLIENT_ID || "",
      clientSecret: process.env.WORLDARENA_HF_CLIENT_SECRET || "",
      authorizeUrl: "https://huggingface.co/oauth/authorize",
      tokenUrl: "https://huggingface.co/oauth/token",
      scope: "openid profile",
      userUrl: "https://huggingface.co/oauth/userinfo"
    }
  };

  function truncate(value, max = 240) {
    const text = String(value || "");
    return text.length > max ? `${text.slice(0, max - 1)}…` : text;
  }

  function parseCookies(request) {
    const header = request.headers.cookie || "";
    const cookies = {};

    header.split(";").forEach((part) => {
      const index = part.indexOf("=");

      if (index === -1) {
        return;
      }

      const key = part.slice(0, index).trim();
      const value = part.slice(index + 1).trim();

      if (key) {
        cookies[key] = decodeURIComponent(value);
      }
    });

    return cookies;
  }

  function signPayload(payload) {
    return crypto.createHmac("sha256", authSecret).update(payload).digest("base64url");
  }

  function encodeSession(session) {
    const body = Buffer.from(JSON.stringify(session), "utf8").toString("base64url");
    const signature = signPayload(body);
    return `${body}.${signature}`;
  }

  function decodeSession(token) {
    if (!token || typeof token !== "string") {
      return null;
    }

    const parts = token.split(".");

    if (parts.length !== 2) {
      return null;
    }

    const [body, signature] = parts;

    if (signPayload(body) !== signature) {
      return null;
    }

    try {
      const session = JSON.parse(Buffer.from(body, "base64url").toString("utf8"));

      if (!session?.id || !session?.expiresAt || Date.now() > Date.parse(session.expiresAt)) {
        return null;
      }

      return session;
    } catch {
      return null;
    }
  }

  function setCookie(response, name, value, maxAgeSeconds = SESSION_TTL_MS / 1000) {
    const parts = [
      `${name}=${encodeURIComponent(value)}`,
      "Path=/",
      "HttpOnly",
      "SameSite=Lax",
      `Max-Age=${maxAgeSeconds}`
    ];

    if (/^https:/i.test(siteUrl)) {
      parts.push("Secure");
    }

    response.setHeader("Set-Cookie", parts.join("; "));
  }

  function clearCookie(response, name) {
    response.setHeader("Set-Cookie", `${name}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0`);
  }

  function createSession(profile) {
    return {
      id: profile.id,
      provider: profile.provider,
      name: profile.name,
      login: profile.login || "",
      avatarUrl: profile.avatarUrl || "",
      email: profile.email || "",
      createdAt: new Date().toISOString(),
      expiresAt: new Date(Date.now() + SESSION_TTL_MS).toISOString()
    };
  }

  function getSessionFromRequest(request) {
    const cookies = parseCookies(request);
    return decodeSession(cookies[SESSION_COOKIE]);
  }

  function attachSession(response, session) {
    setCookie(response, SESSION_COOKIE, encodeSession(session));
  }

  function httpsRequest(targetUrl, options = {}) {
    return new Promise((resolve, reject) => {
      const parsed = new URL(targetUrl);
      const requestOptions = {
        protocol: parsed.protocol,
        hostname: parsed.hostname,
        port: parsed.port || 443,
        path: `${parsed.pathname}${parsed.search}`,
        method: options.method || "GET",
        headers: options.headers || {}
      };

      const req = https.request(requestOptions, (res) => {
        let body = "";

        res.on("data", (chunk) => {
          body += chunk;
        });

        res.on("end", () => {
          resolve({
            status: res.statusCode || 0,
            headers: res.headers,
            body
          });
        });
      });

      req.on("error", reject);

      if (options.body) {
        req.write(options.body);
      }

      req.end();
    });
  }

  function parseFormBody(body) {
    const params = new URLSearchParams(body);
    const result = {};

    params.forEach((value, key) => {
      result[key] = value;
    });

    return result;
  }

  async function exchangeOAuthCode(provider, code) {
    const redirectUri = `${siteUrl}/api/auth/oauth/${provider.id}/callback`;
    const body = new URLSearchParams({
      client_id: provider.clientId,
      client_secret: provider.clientSecret,
      code,
      redirect_uri: redirectUri,
      grant_type: "authorization_code"
    }).toString();

    const tokenResponse = await httpsRequest(provider.tokenUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Accept: "application/json",
        "User-Agent": "WorldAtlas Arena-Site/1.0"
      },
      body
    });

    const tokenPayload = tokenResponse.body.trim().startsWith("{")
      ? JSON.parse(tokenResponse.body)
      : parseFormBody(tokenResponse.body);

    const accessToken = tokenPayload.access_token;

    if (!accessToken) {
      throw new Error("OAuth token exchange failed");
    }

    const userResponse = await httpsRequest(provider.userUrl, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        Accept: "application/json",
        "User-Agent": "WorldAtlas Arena-Site/1.0"
      }
    });

    if (userResponse.status >= 400) {
      throw new Error("OAuth profile fetch failed");
    }

    const profile = JSON.parse(userResponse.body);
    return normalizeOAuthProfile(provider.id, profile);
  }

  function normalizeOAuthProfile(providerId, profile) {
    if (providerId === "github") {
      return {
        id: `github:${profile.id}`,
        provider: "github",
        name: profile.name || profile.login || "GitHub User",
        login: profile.login || "",
        avatarUrl: profile.avatar_url || "",
        email: profile.email || ""
      };
    }

    if (providerId === "google") {
      return {
        id: `google:${profile.sub}`,
        provider: "google",
        name: profile.name || profile.email || "Google User",
        login: profile.email || "",
        avatarUrl: profile.picture || "",
        email: profile.email || ""
      };
    }

    return {
      id: `huggingface:${profile.sub || profile.preferred_username || profile.name}`,
      provider: "huggingface",
      name: profile.name || profile.preferred_username || "Hugging Face User",
      login: profile.preferred_username || profile.sub || "",
      avatarUrl: profile.picture || "",
      email: profile.email || ""
    };
  }

  function readFeedbackEntries() {
    try {
      const payload = JSON.parse(fs.readFileSync(feedbackLogPath, "utf8"));
      return Array.isArray(payload.entries) ? payload.entries : [];
    } catch {
      return [];
    }
  }

  function writeFeedbackEntries(entries) {
    fs.mkdirSync(path.dirname(feedbackLogPath), { recursive: true });
    fs.writeFileSync(feedbackLogPath, JSON.stringify({ entries }, null, 2));
  }

  function sanitizeFeedback(payload, session, request) {
    const category = ["bug", "feature", "arena", "docs", "other"].includes(payload.category)
      ? payload.category
      : "other";
    const message = truncate(String(payload.message || "").trim(), 4000);

    if (message.length < 8) {
      return null;
    }

    return {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      createdAt: new Date().toISOString(),
      category,
      message,
      page: truncate(payload.page || request.headers.referer || "", 240),
      contactEmail: truncate(payload.contactEmail || session?.email || "", 160),
      userId: session?.id || "",
      userName: session?.name || truncate(payload.contactName || "", 120),
      userProvider: session?.provider || "anonymous"
    };
  }

  function authConfigPayload() {
    const providers = Object.values(oauthProviders).map((provider) => ({
      id: provider.id,
      label: provider.label,
      enabled: provider.enabled,
      authorizePath: `/api/auth/oauth/${provider.id}`
    }));
    const enabledProviders = providers.filter((provider) => provider.enabled);

    return {
      siteUrl,
      requireAuthForVotes,
      allowGuestAuth: allowGuestAuth && enabledProviders.length === 0,
      oauthConfigured: enabledProviders.length > 0,
      providers
    };
  }

  function publicSession(session) {
    if (!session) {
      return null;
    }

    return {
      id: session.id,
      provider: session.provider,
      name: session.name,
      login: session.login,
      avatarUrl: session.avatarUrl
    };
  }

  function redirect(response, location) {
    response.writeHead(302, {
      Location: location,
      "Cache-Control": "no-store"
    });
    response.end();
  }

  async function handleAuthFeedbackApi(request, response, pathname, parsedUrl) {
    if (pathname === "/api/auth/config" && request.method === "GET") {
      sendJson(request, response, authConfigPayload());
      return true;
    }

    if (pathname === "/api/auth/session" && request.method === "GET") {
      sendJson(request, response, {
        user: publicSession(getSessionFromRequest(request)),
        config: authConfigPayload()
      });
      return true;
    }

    if (pathname === "/api/auth/logout" && request.method === "POST") {
      clearCookie(response, SESSION_COOKIE);
      sendJson(request, response, { ok: true });
      return true;
    }

    if (pathname === "/api/auth/guest" && request.method === "POST") {
      const enabledProviders = Object.values(oauthProviders).some((provider) => provider.enabled);

      if (!allowGuestAuth || enabledProviders) {
        sendBadRequest(response, "Guest sign-in is disabled when OAuth providers are configured");
        return true;
      }

      try {
        const payload = await parseJsonRequest(request);
        const displayName = truncate(String(payload.displayName || "").trim(), 80);

        if (displayName.length < 2) {
          sendBadRequest(response, "Display name is required");
          return true;
        }

        const session = createSession({
          id: `guest:${crypto.randomUUID()}`,
          provider: "guest",
          name: displayName,
          login: displayName.toLowerCase().replace(/\s+/g, "-"),
          avatarUrl: "",
          email: truncate(payload.email || "", 160)
        });

        attachSession(response, session);
        sendJson(request, response, { ok: true, user: publicSession(session) });
      } catch (error) {
        sendBadRequest(response, error.message || "Invalid JSON");
      }

      return true;
    }

    const oauthStartMatch = pathname.match(/^\/api\/auth\/oauth\/(github|google|huggingface)$/);

    if (oauthStartMatch && request.method === "GET") {
      const provider = oauthProviders[oauthStartMatch[1]];

      if (!provider?.enabled) {
        sendBadRequest(response, "OAuth provider is not configured");
        return true;
      }

      const state = crypto.randomBytes(16).toString("hex");
      const redirectUri = `${siteUrl}/api/auth/oauth/${provider.id}/callback`;
      const authorizeUrl = new URL(provider.authorizeUrl);
      authorizeUrl.searchParams.set("client_id", provider.clientId);
      authorizeUrl.searchParams.set("redirect_uri", redirectUri);
      authorizeUrl.searchParams.set("response_type", "code");
      authorizeUrl.searchParams.set("scope", provider.scope);
      authorizeUrl.searchParams.set("state", state);

      setCookie(response, OAUTH_STATE_COOKIE, state, 600);
      redirect(response, authorizeUrl.toString());
      return true;
    }

    const oauthCallbackMatch = pathname.match(/^\/api\/auth\/oauth\/(github|google|huggingface)\/callback$/);

    if (oauthCallbackMatch && request.method === "GET") {
      const provider = oauthProviders[oauthCallbackMatch[1]];
      const cookies = parseCookies(request);
      const expectedState = cookies[OAUTH_STATE_COOKIE];
      const { code, state, error } = parsedUrl.query || {};

      clearCookie(response, OAUTH_STATE_COOKIE);

      if (error || !code || !state || state !== expectedState || !provider?.enabled) {
        redirect(response, `${siteUrl}/?auth=failed`);
        return true;
      }

      try {
        const profile = await exchangeOAuthCode(provider, code);
        const session = createSession(profile);
        attachSession(response, session);
        redirect(response, `${siteUrl}/arena/?auth=signed-in`);
      } catch (oauthError) {
        console.warn(oauthError);
        redirect(response, `${siteUrl}/?auth=failed`);
      }

      return true;
    }

    if (pathname === "/api/feedback" && request.method === "POST") {
      try {
        const payload = await parseJsonRequest(request);
        const session = getSessionFromRequest(request);
        const entry = sanitizeFeedback(payload, session, request);

        if (!entry) {
          sendBadRequest(response, "Feedback message is too short");
          return true;
        }

        const entries = readFeedbackEntries();
        entries.push(entry);
        writeFeedbackEntries(entries);
        sendJson(request, response, { ok: true, feedback: entry });
      } catch (error) {
        sendBadRequest(response, error.message || "Invalid JSON");
      }

      return true;
    }

    return false;
  }

  return {
    handleAuthFeedbackApi,
    getSessionFromRequest,
    publicSession,
    authConfigPayload,
    requireAuthForVotes
  };
}

module.exports = {
  createAuthModule
};
