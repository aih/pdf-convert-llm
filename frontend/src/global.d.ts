declare global {
  namespace React {
    // Corrected: "interface" keyword was missing a space
    interface InputHTMLAttributes<T extends EventTarget> extends HTMLAttributes<T> {
      directory?: string;
      webkitdirectory?: string;
    }
  }
}
export {};