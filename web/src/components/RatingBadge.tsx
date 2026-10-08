import { Badge, type BadgeTone } from "./ui";

const TONES: Record<string, BadgeTone> = { Buy: "gain", Hold: "warn", Sell: "loss" };

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
  return (
    <Badge tone={TONES[rating] ?? "neutral"} size={size}>
      {rating}
    </Badge>
  );
}
