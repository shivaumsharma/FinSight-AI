const COLORS: Record<string, string> = {
  Buy: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  Hold: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  Sell: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  "Insufficient Data": "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300",
};

export function ratingTextColor(rating: string): string {
  switch (rating) {
    case "Buy":
      return "text-green-600 dark:text-green-400";
    case "Hold":
      return "text-amber-600 dark:text-amber-400";
    case "Sell":
      return "text-red-600 dark:text-red-400";
    default:
      return "text-gray-500 dark:text-gray-400";
  }
}

export default function RatingBadge({ rating }: { rating: string }) {
  const cls = COLORS[rating] || COLORS["Insufficient Data"];
  return (
    <span className={`inline-block rounded-full px-4 py-1 text-lg font-semibold ${cls}`}>
      {rating}
    </span>
  );
}
