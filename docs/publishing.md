# Publishing checklist

The repository is private. Before changing its visibility to public, an administrator should work through this list.

## 1. Secrets and private data

- [ ] Search the working tree **and full commit history** for keys, tokens, passwords, and `.env` files (for example with `gitleaks detect` or GitHub secret scanning).
- [ ] Revoke and rotate anything found; rewriting history is not enough on its own.
- [ ] Confirm no personal data, private datasets, or proprietary code is present.

## 2. Licenses

- [ ] `LICENSE` is correct for the original code.
- [ ] Every dependency's license allows the intended use.
- [ ] Every bundled asset, model, or dataset has a compatible license, recorded in the README or alongside the file.

## 3. Content

- [ ] README describes only implemented features, and labels future work as such.
- [ ] No unsupported accuracy, performance, or adoption claims.
- [ ] Docs, examples, and `CONTRIBUTING.md` are up to date.

## 4. Repository settings

- [ ] CI is passing on `main`.
- [ ] Branch protection on `main` (require PR review and passing checks).
- [ ] Private vulnerability reporting enabled (**Settings → Code security**).
- [ ] Repository description, topics, and starter labels set.

## 5. Change visibility

**Settings → General → Danger Zone → Change repository visibility.** This exposes all branches and history, so finish the steps above first.
