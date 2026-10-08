import { notFound } from "next/navigation";
import BenchClient from "./BenchClient";

// Render benchmark harness. Only exists when the server is started with NEXT_PUBLIC_BENCH=1 (see bench/).
export default function BenchPage() {
  if (process.env.NEXT_PUBLIC_BENCH !== "1") notFound();
  return <BenchClient />;
}
