// Tailwind v4's PostCSS plugin is the single dep in 15+. The legacy
// `tailwindcss` PostCSS plugin from v3 is a no-op in v4; keep the
// imports clean so the brand preview compiles fast.
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
