"use client";

import { useEffect } from "react";

const THEME_MENU_SELECTOR =
  '.fern-header-navbar-links button[aria-haspopup="menu"]:not([aria-label])';

function labelThemeMenu(): void {
  document.querySelectorAll<HTMLButtonElement>(THEME_MENU_SELECTOR).forEach((button) => {
    button.setAttribute("aria-label", "Select color theme");
  });
}

/** Adds an accessible name to Fern's icon-only desktop theme menu. */
export function AccessibilityEnhancements() {
  useEffect(() => {
    labelThemeMenu();

    const observer = new MutationObserver(labelThemeMenu);
    observer.observe(document.body, { childList: true, subtree: true });

    return () => observer.disconnect();
  }, []);

  return null;
}
