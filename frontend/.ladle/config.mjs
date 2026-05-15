/** @type {import("@ladle/react").UserConfig} */
export default {
  stories: "src/**/*.stories.{ts,tsx}",
  appendToHead: "",
  defaultStory: "primitives--button-variants",
  addons: {
    a11y: { enabled: true },
    theme: {
      enabled: true,
      defaultState: "light",
    },
    width: { enabled: true, defaultState: 0 },
    rtl: { enabled: true, defaultState: false },
    source: { enabled: true, defaultState: false },
  },
};
