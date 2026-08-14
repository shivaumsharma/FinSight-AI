"use client";

import { useEffect, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
  createChart,
} from "lightweight-charts";

const RANGES = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"] as const;
type Range = (typeof RANGES)[number];
const RANGE_LABELS: Record<Range, string> = {
  "1mo": "1M", "3mo": "3M", "6mo": "6M", "1y": "1Y", "2y": "2Y", "5y": "5Y", max: "MAX",
};

interface PriceBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
}

interface OverlayPoint extends Array<string | number | null> {
  0: string;
  1: number | null;
}

interface TechnicalsResponse {
  price_history: PriceBar[];
  overlays: Record<"ema_20" | "ema_50" | "ema_200" | "bollinger_upper" | "bollinger_lower" | "vwap", OverlayPoint[]>;
}

type OverlayKey = "ema_20" | "ema_50" | "ema_200" | "bollinger" | "vwap";

const OVERLAY_TOGGLES: { key: OverlayKey; label: string }[] = [
  { key: "ema_20", label: "EMA 20" },
  { key: "ema_50", label: "EMA 50" },
  { key: "ema_200", label: "EMA 200" },
  { key: "bollinger", label: "Bollinger" },
  { key: "vwap", label: "VWAP" },
];

function useThemeColor(name: string, fallback: string) {
  const [color, setColor] = useState(fallback);
  useEffect(() => {
    setColor(getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback);
  }, [name, fallback]);
  return color;
}

// Fetches its own /technicals data (same self-fetching convention as
// FinancialPerformanceChart) since the range selector needs to
// re-fetch a different daily-bar window in place. lightweight-charts
// (TradingView's OSS library) is imperative, not React-declarative --
// the chart/series objects are created once in a ref and mutated via
// setData on data/theme changes, not re-rendered as JSX.
export default function PriceChart({ ticker, currency }: { ticker: string; currency: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const overlaySeriesRef = useRef<Partial<Record<OverlayKey, ISeriesApi<"Line">[]>>>({});

  const [range, setRange] = useState<Range>("1y");
  const [data, setData] = useState<TechnicalsResponse | null>(null);
  const [error, setError] = useState(false);
  const [visibleOverlays, setVisibleOverlays] = useState<Set<OverlayKey>>(new Set(["ema_20"]));

  const accent = useThemeColor("--accent", "#00d97e");
  const danger = useThemeColor("--danger", "#ff4d4f");
  const dim = useThemeColor("--dim", "#3a4353");
  const textColor = useThemeColor("--muted", "#9aa4b2");
  const borderColor = useThemeColor("--border-subtle", "#131a22");

  useEffect(() => {
    setData(null);
    setError(false);
    fetch(`/api/stock/${encodeURIComponent(ticker)}/technicals?range=${range}`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setError(true));
  }, [ticker, range]);

  // Chart creation -- once per mount, torn down on unmount. Colors are
  // read fresh each time this effect re-runs (theme change), which
  // recreates the chart -- simplest correct way to keep it theme-aware
  // without lightweight-charts' own more granular applyOptions dance.
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor, fontFamily: "var(--font-mono)" },
      grid: { vertLines: { color: borderColor }, horzLines: { color: borderColor } },
      rightPriceScale: { borderColor },
      timeScale: { borderColor, timeVisible: false },
      height: 320,
      autoSize: true,
    });
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: accent, downColor: danger, borderVisible: false, wickUpColor: accent, wickDownColor: danger,
    });
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" }, priceScaleId: "volume",
    });
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    return () => {
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      overlaySeriesRef.current = {};
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accent, danger, textColor, borderColor]);

  // Data + overlay updates -- separate from chart creation so toggling
  // an overlay checkbox doesn't tear down and recreate the whole chart.
  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    const volumeSeries = volumeSeriesRef.current;
    if (!chart || !candleSeries || !volumeSeries || !data) return;

    candleSeries.setData(
      data.price_history.map((bar) => ({ time: bar.date as unknown as UTCTimestamp, open: bar.open, high: bar.high, low: bar.low, close: bar.close }))
    );
    volumeSeries.setData(
      data.price_history.map((bar, i) => ({
        time: bar.date as unknown as UTCTimestamp,
        value: bar.volume ?? 0,
        color: i > 0 && bar.close < data.price_history[i - 1].close ? danger : accent,
      }))
    );

    // Tear down every existing overlay series, then re-add only the
    // ones currently toggled on -- simpler and safer than diffing
    // which overlays changed, and cheap (at most 6 line series).
    for (const seriesList of Object.values(overlaySeriesRef.current)) {
      seriesList?.forEach((s) => chart.removeSeries(s));
    }
    overlaySeriesRef.current = {};

    const addLine = (points: OverlayPoint[], color: string, lineWidth: 1 | 2 = 1) => {
      const series = chart.addSeries(LineSeries, { color, lineWidth, priceLineVisible: false, lastValueVisible: false });
      series.setData(
        points.filter((p) => p[1] !== null).map((p) => ({ time: p[0] as unknown as UTCTimestamp, value: p[1] as number }))
      );
      return series;
    };

    if (visibleOverlays.has("ema_20")) overlaySeriesRef.current.ema_20 = [addLine(data.overlays.ema_20, accent)];
    if (visibleOverlays.has("ema_50")) overlaySeriesRef.current.ema_50 = [addLine(data.overlays.ema_50, "#e0a92f")];
    if (visibleOverlays.has("ema_200")) overlaySeriesRef.current.ema_200 = [addLine(data.overlays.ema_200, danger)];
    if (visibleOverlays.has("vwap")) overlaySeriesRef.current.vwap = [addLine(data.overlays.vwap, "#5b8def")];
    if (visibleOverlays.has("bollinger")) {
      overlaySeriesRef.current.bollinger = [
        addLine(data.overlays.bollinger_upper, dim),
        addLine(data.overlays.bollinger_lower, dim),
      ];
    }

    chart.timeScale().fitContent();
  }, [data, visibleOverlays, accent, danger, dim]);

  function toggleOverlay(key: OverlayKey) {
    setVisibleOverlays((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-mono text-[10px] tracking-wide text-dim">PRICE CHART</p>
        <div className="flex gap-1 rounded-lg border border-border bg-card p-0.5">
          {RANGES.map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => setRange(r)}
              className={`rounded px-2 py-1 font-mono text-[10px] font-bold ${
                range === r ? "bg-accent text-bg" : "text-muted hover:text-text"
              }`}
            >
              {RANGE_LABELS[r]}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-1.5">
        {OVERLAY_TOGGLES.map((o) => (
          <button
            key={o.key}
            type="button"
            onClick={() => toggleOverlay(o.key)}
            className={`rounded border px-2 py-0.5 font-mono text-[10px] font-bold ${
              visibleOverlays.has(o.key) ? "border-accent text-accent" : "border-border text-dim hover:text-muted"
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>

      <div className="mt-2 rounded-lg border border-border bg-card px-2 py-2">
        {error ? (
          <p className="py-10 text-center font-mono text-[11px] text-dim">Couldn&apos;t load price history for this ticker.</p>
        ) : !data ? (
          <div className="h-[320px] animate-pulse rounded bg-card/60" />
        ) : null}
        <div ref={containerRef} className={data && !error ? "h-[320px] w-full" : "hidden"} />
      </div>
      <p className="mt-1 font-mono text-[9px] text-dim">Currency: {currency}. Daily bars -- intraday (1D/1W) ranges aren&apos;t available.</p>
    </div>
  );
}
