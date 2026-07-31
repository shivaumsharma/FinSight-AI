const COLORS: Record<string, string> = {
  Buy: "text-accent border-accent",
  Hold: "text-warn border-warn",
  Sell: "text-danger border-danger",
  "Insufficient Data": "text-muted border-dim",
};

export function ratingColorClass(rating: string): string {
  switch (rating) {
    case "Buy":
      return "text-accent";
    case "Hold":
      return "text-warn";
    case "Sell":
      return "text-danger";
    default:
      return "text-muted";
  }
}

export default function RatingBadge({ rating, size = "md" }: { rating: string; size?: "sm" | "md" }) {
  const cls = COLORS[rating] || COLORS["Insufficient Data"];
  const sizing = size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-3 py-1 text-sm";
  return (
    <span className={`inline-block rounded font-mono font-bold uppercase tracking-wide border ${cls} ${sizing}`}>
      {rating}
    </span>
  );
}
