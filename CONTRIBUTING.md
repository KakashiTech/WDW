# Contributing to WDW.jl

## How to Contribute

1. **Fork** the repo and create a feature branch from `main`.
2. **Run tests** before any commit: `julia --project -e 'using Pkg; Pkg.test()'`
3. **Add tests** for any new functionality. All tests must pass.
4. **Document** public functions with docstrings.
5. **Open a PR** with a clear description of changes.

## Code Style

- Follow Julia's style guide (https://docs.julialang.org/en/v1/manual/style-guide/)
- No comments in code (documentation should speak for itself)
- Export only the public API; internal functions stay unexported
- Use `using WDW.ModuleName: InternalFunction` for internal access

## Testing

Tests run in two modes:
- **DEMO** (default): `julia --project -e 'using Pkg; Pkg.test()'` — runs core tests
- **FULL**: `FULL=1 julia --project -e 'using Pkg; Pkg.test()'` — runs all tests including extras/

All contributions must pass DEMO mode at minimum.

## Commit Messages

Use conventional commits: `type(scope): message`
- `fix(bispectrum): correct odd-n conjugate indexing`
- `feat(pipeline): add sheaf-based cumulative statistics`
- `docs(readme): update module architecture diagram`
- `test(rupture): add reproducibility across seeds`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
