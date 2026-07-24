const authUiText = {
  en: {
    signIn: "Sign In",
    signOut: "Sign Out",
    signedInAs: "Signed in as",
    feedback: "Feedback",
    authTitle: "Sign in (optional)",
    authBody: "You can browse the site without signing in. Sign in with a real account to submit Arena Battle votes and attach your identity to human preference research data.",
    continueGithub: "Continue with GitHub",
    continueGoogle: "Continue with Google",
    continueHf: "Continue with Hugging Face",
    authDismiss: "Not now",
    oauthSetupHint: "OAuth is not configured on this server. Set WORLDARENA_*_CLIENT_ID and WORLDARENA_*_CLIENT_SECRET in site/.env, then restart the site server. See site/AUTH.md.",
    oauthUnavailable: "This provider is not configured on the server yet.",
    authRequired: "Sign in to start voting on video world models.",
    authRequiredSubmit: "Sign in before submitting your scorecard.",
    authFailed: "Sign in failed. Please try again.",
    authSignedIn: "Signed in successfully.",
    feedbackTitle: "Send feedback",
    feedbackBody: "Report bugs, suggest features, or tell us how the arena and docs can improve.",
    feedbackCategory: "Category",
    feedbackMessage: "Message",
    feedbackEmail: "Contact email (optional)",
    feedbackSubmit: "Send feedback",
    feedbackSending: "Sending…",
    feedbackSent: "Thanks — your feedback was saved.",
    feedbackFailed: "Could not send feedback. Please try again.",
    feedbackCategories: {
      bug: "Bug",
      feature: "Feature request",
      arena: "Arena battle",
      docs: "Documentation",
      other: "Other"
    }
  },
  zh: {
    signIn: "登录",
    signOut: "退出",
    signedInAs: "当前用户",
    feedback: "反馈",
    authTitle: "登录（可选）",
    authBody: "不登录也可以浏览站点。使用真实账号登录后，方可提交 Arena Battle 投票，并将身份关联到人类偏好研究数据。",
    continueGithub: "使用 GitHub 继续",
    continueGoogle: "使用 Google 继续",
    continueHf: "使用 Hugging Face 继续",
    authDismiss: "暂不登录",
    oauthSetupHint: "当前服务器尚未配置 OAuth。请在 site/.env 中设置 WORLDARENA_*_CLIENT_ID 与 WORLDARENA_*_CLIENT_SECRET，然后重启站点服务。详见 site/AUTH.md。",
    oauthUnavailable: "该登录方式尚未在服务器上配置。",
    authRequired: "登录后即可开始对视频世界模型投票。",
    authRequiredSubmit: "提交打分表前请先登录。",
    authFailed: "登录失败，请重试。",
    authSignedIn: "登录成功。",
    feedbackTitle: "提交反馈",
    feedbackBody: "报告问题、提出功能建议，或告诉我们 Arena 与文档如何改进。",
    feedbackCategory: "类别",
    feedbackMessage: "留言",
    feedbackEmail: "联系邮箱（可选）",
    feedbackSubmit: "发送反馈",
    feedbackSending: "发送中…",
    feedbackSent: "感谢，反馈已保存。",
    feedbackFailed: "反馈发送失败，请重试。",
    feedbackCategories: {
      bug: "Bug",
      feature: "功能建议",
      arena: "Arena 对局",
      docs: "文档",
      other: "其他"
    }
  }
};

const authProviderOrder = ["github", "google", "huggingface"];

let authConfig = {
  requireAuthForVotes: true,
  allowGuestAuth: false,
  oauthConfigured: false,
  providers: authProviderOrder.map((id) => ({
    id,
    label: id === "github" ? "GitHub" : id === "google" ? "Google" : "Hugging Face",
    enabled: false,
    authorizePath: `/api/auth/oauth/${id}`
  }))
};
let authUser = null;
let authUiLanguage = "en";
let authFeedbackBound = false;

function notifyAuthChanged() {
  document.dispatchEvent(new CustomEvent("worldarena-auth-changed"));
}

export function setAuthUiLanguage(language) {
  authUiLanguage = language === "zh" ? "zh" : "en";
}

function authText(key) {
  const lang = authUiText[authUiLanguage] || authUiText.en;
  return lang[key] || authUiText.en[key] || key;
}

function feedbackCategoryLabel(key) {
  const lang = authUiText[authUiLanguage] || authUiText.en;
  return lang.feedbackCategories[key] || authUiText.en.feedbackCategories[key] || key;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/'/g, "&#39;");
}

export function getAuthUser() {
  return authUser;
}

export function getAuthConfig() {
  return authConfig;
}

export function isAuthRequiredForVotes() {
  return Boolean(authConfig.requireAuthForVotes);
}

export function isSignedIn() {
  return Boolean(authUser);
}

export async function refreshAuthSession() {
  try {
    const response = await fetch("/api/auth/session", { credentials: "same-origin" });

    if (!response.ok) {
      throw new Error(`Auth session failed: ${response.status}`);
    }

    const payload = await response.json();
    authUser = payload.user || null;
    authConfig = payload.config || authConfig;
    updateAuthUi();
    notifyAuthChanged();
    return authUser;
  } catch (error) {
    console.warn(error);
    authUser = null;
    updateAuthUi();
    return null;
  }
}

function authProviderLabel(provider) {
  if (provider.id === "github") {
    return authText("continueGithub");
  }

  if (provider.id === "google") {
    return authText("continueGoogle");
  }

  if (provider.id === "huggingface") {
    return authText("continueHf");
  }

  return provider.label || provider.id;
}

function authProviderButtonsMarkup() {
  const providersById = new Map(
    (Array.isArray(authConfig.providers) ? authConfig.providers : []).map((provider) => [provider.id, provider])
  );

  return authProviderOrder
    .map((providerId) => {
      const provider = providersById.get(providerId) || {
        id: providerId,
        enabled: false,
        authorizePath: `/api/auth/oauth/${providerId}`
      };
      const label = authProviderLabel(provider);
      const authorizePath = provider.authorizePath || `/api/auth/oauth/${provider.id}`;

      if (provider.enabled) {
        return `<a class="auth-provider-button" href="${escapeAttr(authorizePath)}" data-auth-provider="${escapeAttr(provider.id)}">${escapeHtml(label)}</a>`;
      }

      return `<button type="button" class="auth-provider-button auth-provider-button-disabled" data-auth-provider-disabled="${escapeAttr(provider.id)}" aria-disabled="true" title="${escapeAttr(authText("oauthUnavailable"))}">${escapeHtml(label)}</button>`;
    })
    .join("");
}

function authSetupHintMarkup() {
  if (authConfig.oauthConfigured) {
    return "";
  }

  return `<p class="auth-oauth-setup-hint">${escapeHtml(authText("oauthSetupHint"))}</p>`;
}

export function authHeaderActionsMarkup() {
  return `
    <div class="auth-user-chip" data-auth-user-chip hidden>
      <button type="button" class="auth-user-button" data-auth-user-button>
        <span class="auth-user-avatar" data-auth-user-avatar aria-hidden="true"></span>
        <span data-auth-user-name></span>
      </button>
      <div class="auth-user-menu" data-auth-user-menu hidden>
        <p><span>${escapeHtml(authText("signedInAs"))}</span> <strong data-auth-user-menu-name></strong></p>
        <button type="button" data-auth-sign-out>${escapeHtml(authText("signOut"))}</button>
      </div>
    </div>
    <button type="button" class="header-utility-button" data-auth-open hidden>${escapeHtml(authText("signIn"))}</button>
    <button type="button" class="header-utility-button" data-feedback-open>${escapeHtml(authText("feedback"))}</button>
  `;
}

export function authFeedbackModalsMarkup() {
  return `
    <div class="site-modal" data-auth-modal hidden>
      <button type="button" class="site-modal-scrim" data-auth-close aria-label="Close"></button>
      <section class="site-modal-panel auth-modal-panel" role="dialog" aria-modal="true" aria-label="${escapeAttr(authText("authTitle"))}">
        <header>
          <h2>${escapeHtml(authText("authTitle"))}</h2>
          <p>${escapeHtml(authText("authBody"))}</p>
        </header>
        <div class="auth-provider-list" data-auth-provider-list>
          ${authProviderButtonsMarkup()}
        </div>
        <div data-auth-setup-hint>${authSetupHintMarkup()}</div>
        <footer class="auth-modal-footer">
          <button type="button" class="auth-dismiss-button" data-auth-close>${escapeHtml(authText("authDismiss"))}</button>
        </footer>
        <p class="site-modal-status" data-auth-status hidden></p>
      </section>
    </div>

    <div class="site-modal" data-feedback-modal hidden>
      <button type="button" class="site-modal-scrim" data-feedback-close aria-label="Close"></button>
      <section class="site-modal-panel feedback-modal-panel" role="dialog" aria-modal="true" aria-label="${escapeAttr(authText("feedbackTitle"))}">
        <header>
          <h2>${escapeHtml(authText("feedbackTitle"))}</h2>
          <p>${escapeHtml(authText("feedbackBody"))}</p>
        </header>
        <form data-feedback-form>
          <label>
            <span>${escapeHtml(authText("feedbackCategory"))}</span>
            <select name="category">
              ${Object.keys(authUiText.en.feedbackCategories)
                .map((key) => `<option value="${escapeAttr(key)}">${escapeHtml(feedbackCategoryLabel(key))}</option>`)
                .join("")}
            </select>
          </label>
          <label>
            <span>${escapeHtml(authText("feedbackMessage"))}</span>
            <textarea name="message" rows="6" maxlength="4000" required></textarea>
          </label>
          <label>
            <span>${escapeHtml(authText("feedbackEmail"))}</span>
            <input type="email" name="contactEmail" maxlength="160" autocomplete="email" />
          </label>
          <button type="submit">${escapeHtml(authText("feedbackSubmit"))}</button>
        </form>
        <p class="site-modal-status" data-feedback-status hidden></p>
      </section>
    </div>
  `;
}

export function authFeedbackShellMarkup() {
  return `${authHeaderActionsMarkup()}${authFeedbackModalsMarkup()}`;
}

function updateAuthUi() {
  const chip = document.querySelector("[data-auth-user-chip]");
  const openButton = document.querySelector("[data-auth-open]");
  const avatar = document.querySelector("[data-auth-user-avatar]");
  const nameNode = document.querySelector("[data-auth-user-name]");
  const menuName = document.querySelector("[data-auth-user-menu-name]");
  const providerList = document.querySelector("[data-auth-provider-list]");
  const setupHint = document.querySelector("[data-auth-setup-hint]");

  if (providerList) {
    providerList.innerHTML = authProviderButtonsMarkup();
  }

  if (setupHint) {
    setupHint.innerHTML = authSetupHintMarkup();
  }

  if (authUser) {
    chip?.removeAttribute("hidden");
    openButton?.setAttribute("hidden", "hidden");

    if (nameNode) {
      nameNode.textContent = authUser.name || authUser.login || "User";
    }

    if (menuName) {
      menuName.textContent = authUser.name || authUser.login || "User";
    }

    if (avatar) {
      if (authUser.avatarUrl) {
        avatar.style.backgroundImage = `url("${authUser.avatarUrl}")`;
        avatar.textContent = "";
      } else {
        avatar.style.backgroundImage = "";
        avatar.textContent = (authUser.name || "U").slice(0, 1).toUpperCase();
      }
    }
  } else {
    chip?.setAttribute("hidden", "hidden");
    openButton?.removeAttribute("hidden");
  }
}

export function openAuthModal(message = "") {
  const modal = document.querySelector("[data-auth-modal]");
  const status = document.querySelector("[data-auth-status]");

  if (!modal) {
    return;
  }

  modal.removeAttribute("hidden");
  document.body.classList.add("modal-open");

  if (status) {
    if (message) {
      status.textContent = message;
      status.removeAttribute("hidden");
    } else {
      status.textContent = "";
      status.setAttribute("hidden", "hidden");
    }
  }
}

export function closeAuthModal() {
  document.querySelector("[data-auth-modal]")?.setAttribute("hidden", "hidden");
  document.body.classList.remove("modal-open");
}

export function openFeedbackModal() {
  const modal = document.querySelector("[data-feedback-modal]");
  const status = document.querySelector("[data-feedback-status]");

  if (!modal) {
    return;
  }

  modal.removeAttribute("hidden");
  document.body.classList.add("modal-open");

  if (status) {
    status.textContent = "";
    status.setAttribute("hidden", "hidden");
  }

  modal.querySelector("textarea")?.focus();
}

export function closeFeedbackModal() {
  document.querySelector("[data-feedback-modal]")?.setAttribute("hidden", "hidden");
  document.body.classList.remove("modal-open");
}

export function requireSignedInForArena(message) {
  if (!isAuthRequiredForVotes() || isSignedIn()) {
    return true;
  }

  openAuthModal(message || authText("authRequiredSubmit"));
  return false;
}

export function bindAuthFeedbackUi() {
  if (authFeedbackBound) {
    updateAuthUi();
    return;
  }

  authFeedbackBound = true;

  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-auth-provider-disabled]")) {
      openAuthModal(authText("oauthUnavailable"));
      return;
    }

    if (event.target.closest("[data-auth-open]")) {
      openAuthModal();
      return;
    }

    if (event.target.closest("[data-feedback-open]")) {
      openFeedbackModal();
      return;
    }

    if (event.target.closest("[data-auth-close]")) {
      closeAuthModal();
      return;
    }

    if (event.target.closest("[data-feedback-close]")) {
      closeFeedbackModal();
      return;
    }

    if (event.target.closest("[data-auth-user-button]")) {
      document.querySelector("[data-auth-user-menu]")?.toggleAttribute("hidden");
      return;
    }

    if (event.target.closest("[data-auth-sign-out]")) {
      fetch("/api/auth/logout", { method: "POST", credentials: "same-origin" })
        .finally(() => refreshAuthSession())
        .finally(() => {
          document.querySelector("[data-auth-user-menu]")?.setAttribute("hidden", "hidden");
        });
    }
  });

  document.addEventListener("submit", async (event) => {
    const feedbackForm = event.target.closest("[data-feedback-form]");

    if (!feedbackForm) {
      return;
    }

    event.preventDefault();
    const status = document.querySelector("[data-feedback-status]");
    const submitButton = feedbackForm.querySelector('button[type="submit"]');
    const formData = new FormData(feedbackForm);

    if (submitButton) {
      submitButton.disabled = true;
      submitButton.textContent = authText("feedbackSending");
    }

    try {
      const response = await fetch("/api/feedback", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          category: formData.get("category"),
          message: formData.get("message"),
          contactEmail: formData.get("contactEmail"),
          page: window.location.pathname + window.location.search
        })
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || authText("feedbackFailed"));
      }

      feedbackForm.reset();

      if (status) {
        status.textContent = authText("feedbackSent");
        status.removeAttribute("hidden");
      }

      window.setTimeout(() => closeFeedbackModal(), 900);
    } catch (error) {
      if (status) {
        status.textContent = error.message || authText("feedbackFailed");
        status.removeAttribute("hidden");
      }
    } finally {
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent = authText("feedbackSubmit");
      }
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }

    closeAuthModal();
    closeFeedbackModal();
  });

  const params = new URLSearchParams(window.location.search);

  if (params.get("auth") === "failed") {
    openAuthModal(authText("authFailed"));
    params.delete("auth");
    const next = `${window.location.pathname}${params.toString() ? `?${params.toString()}` : ""}`;
    window.history.replaceState({}, "", next);
  } else if (params.get("auth") === "signed-in") {
    openAuthModal(authText("authSignedIn"));
    params.delete("auth");
    const next = `${window.location.pathname}${params.toString() ? `?${params.toString()}` : ""}`;
    window.history.replaceState({}, "", next);
    window.setTimeout(() => closeAuthModal(), 900);
  }
}

export function authTextForKey(key) {
  return authText(key);
}
