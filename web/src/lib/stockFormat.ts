// Stock-detail-page helpers. Thin wrappers over numberFormat.ts, kept so existing call sites and signatures still work.

import { formatCompact, formatCompactMoney, formatPercent, formatPrice, formatRatio } from "./numberFormat";

export const fmtCompactNumber = (n: number | null | undefined): string => formatCompact(n);

export const fmtCompactMoney = (n: number | null | undefined, symbol: string): string => formatCompactMoney(n, symbol);

export const fmtPrice = (n: number | null | undefined, symbol: string): string => formatPrice(n, symbol);

export const fmtPercent = (n: number | null | undefined, alreadyFraction = false): string =>
  formatPercent(n, { fraction: alreadyFraction });

export const fmtRatio = (n: number | null | undefined, suffix = "x"): string => formatRatio(n, suffix);
