// Pulls @testing-library/jest-dom matchers into the global expect.
// Imported once here so every test file picks them up via Vitest's
// `setupFiles` hook (no per-file boilerplate).
import "@testing-library/jest-dom/vitest";
