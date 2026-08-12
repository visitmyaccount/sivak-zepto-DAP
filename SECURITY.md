# Public Repository Safety Rules

This is a public learning repository. The following rules apply to every branch and commit:

1. Never commit passwords, tokens, API keys, private keys, or other credentials.
2. Never commit a real environment file. Runtime secrets must be supplied through the shell or a hosting platform's secret settings.
3. Documentation and sample configuration must contain placeholders only.
4. Do not add automated-author branding or generated-by attribution to source files, documentation, commits, branches, or pull requests.
5. Inspect staged filenames and content before every commit, and inspect the full history before the final push.
6. Keep local caches, virtual environments, downloaded models, and ChromaDB runtime data outside version control.

If a secret is staged accidentally, stop before committing, unstage the file, remove the value, and rotate the credential if it may have been exposed.
