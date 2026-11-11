# Security Policy

## Supported versions

AI Learning Lab is an educational project in early development. Security fixes target the latest commit on `main`.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Use GitHub's private vulnerability reporting (**Security → Report a vulnerability**) if it is enabled on the repository, or contact the maintainer privately through their GitHub profile.

Include what you found, how to reproduce it, and its likely impact. We aim to acknowledge reports promptly; as a volunteer project, we cannot guarantee a response time.

## Guidance for users and contributors

- The MVP makes no network calls and needs no credentials. Keep it that way for core examples.
- Never commit secrets, API keys, tokens, or personal data. If one is committed by mistake, revoke it first; removing it from history alone is not enough.
- Loading pickled or joblib model files can execute arbitrary code. Only load model files from sources you trust.
- Treat third-party models and datasets as untrusted input and review their licenses and provenance before use.
- This project is not designed for production or safety-critical decisions.
