import type { ButtonHTMLAttributes } from "react";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

const VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-accent text-bg hover:opacity-90",
  danger: "bg-danger text-bg hover:opacity-90",
  secondary: "border border-border bg-card text-muted hover:border-accent hover:text-accent",
  ghost: "text-muted hover:text-accent",
};

// Both sizes keep the hit area at least 32px tall (WCAG 2.2 asks for 24px).
const SIZES: Record<ButtonSize, string> = {
  sm: "min-h-8 px-2.5 text-micro",
  md: "min-h-9 px-card-x text-small",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  fullWidth?: boolean;
  // Disables the button and marks it busy; pair with a changed label such as "Placing order...".
  loading?: boolean;
}

export default function Button({
  variant = "secondary",
  size = "md",
  fullWidth = false,
  loading = false,
  type = "button",
  disabled,
  className = "",
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={`inline-flex items-center justify-center rounded-control font-mono font-bold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-40 ${VARIANTS[variant]} ${SIZES[size]} ${fullWidth ? "w-full" : ""} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}
