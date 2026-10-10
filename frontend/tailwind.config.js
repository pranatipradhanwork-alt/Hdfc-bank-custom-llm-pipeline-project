/** HDFC AI Platform theme: brand red, neutral "ink" greys, Inter for text and Manrope for headings. */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: { 50: "#FDF3F3", 100: "#FBE4E5", 200: "#F5C2C5", 300: "#EA969B", 400: "#DD6169", 500: "#C81E3A", 600: "#AD1730", 700: "#8C1227", 800: "#6E0F20", 900: "#560D1A" },
        ink: { 25: "#FBFBFC", 50: "#F6F6F8", 100: "#EEEEF1", 200: "#DDDDE3", 300: "#B9B9C3", 400: "#8C8C99", 500: "#6B6B78", 600: "#4E4E5A", 700: "#383842", 800: "#24242C", 900: "#15151B" },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["Manrope", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(21,21,27,0.04), 0 1px 2px rgba(21,21,27,0.06)",
        hover: "0 8px 20px rgba(21,21,27,0.10)",
      },
    },
  },
  plugins: [],
};
