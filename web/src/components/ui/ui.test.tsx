import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Badge, Button, ChartFrame, Table } from "./index";

describe("Button", () => {
  it("is a non-submitting button by default", () => {
    render(<Button>Go</Button>);
    expect(screen.getByRole("button", { name: "Go" })).toHaveAttribute("type", "button");
  });

  it("can submit a form when asked to", () => {
    render(<Button type="submit">Send</Button>);
    expect(screen.getByRole("button", { name: "Send" })).toHaveAttribute("type", "submit");
  });

  it("fires onClick, and not when disabled", () => {
    const onClick = vi.fn();
    const { rerender } = render(<Button onClick={onClick}>Go</Button>);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledTimes(1);
    rerender(<Button onClick={onClick} disabled>Go</Button>);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("is disabled and marked busy while loading", () => {
    render(<Button loading>Saving</Button>);
    const b = screen.getByRole("button");
    expect(b).toBeDisabled();
    expect(b).toHaveAttribute("aria-busy", "true");
  });

  it("applies the variant and size classes", () => {
    render(<Button variant="danger" size="sm">Sell</Button>);
    const cls = screen.getByRole("button").className;
    expect(cls).toContain("bg-danger");
    expect(cls).toContain("min-h-8");
  });

  it("keeps every size at least 32px tall", () => {
    for (const size of ["sm", "md"] as const) {
      const { unmount } = render(<Button size={size}>x</Button>);
      expect(screen.getByRole("button").className).toMatch(/min-h-(8|9)\b/);
      unmount();
    }
  });
});

describe("Badge", () => {
  it.each([
    ["gain", "text-accent"],
    ["warn", "text-warn"],
    ["loss", "text-danger"],
    ["neutral", "text-muted"],
  ] as const)("%s tone", (tone, cls) => {
    render(<Badge tone={tone}>Label</Badge>);
    expect(screen.getByText("Label").className).toContain(cls);
  });
});

describe("Table", () => {
  const columns = ["Metric", "AAPL", "MSFT"];

  it("renders headers with scope and right-aligns value columns", () => {
    render(<Table columns={columns} rows={[{ key: "pe", label: "P/E", cells: ["28.5x", "34.1x"] }]} />);
    expect(screen.getByRole("columnheader", { name: "AAPL" })).toHaveAttribute("scope", "col");
    expect(screen.getByRole("rowheader", { name: "P/E" })).toBeInTheDocument();
    expect(screen.getByText("28.5x").className).toContain("text-right");
  });

  it("tones a cell when given a value object, and leaves plain cells default", () => {
    render(
      <Table
        columns={columns}
        rows={[{ key: "g", label: "Growth", cells: [{ value: "+5.0%", tone: "gain" }, { value: "-2.0%", tone: "loss" }, "n/a"] }]}
      />,
    );
    expect(screen.getByText("+5.0%").className).toContain("text-accent");
    expect(screen.getByText("-2.0%").className).toContain("text-danger");
    expect(screen.getByText("n/a").className).toContain("text-text");
  });

  it("accepts React nodes as cells", () => {
    render(<Table columns={columns} rows={[{ key: "l", label: "Link", cells: [<a key="a" href="/x">open</a>, "-"] }]} />);
    expect(screen.getByRole("link", { name: "open" })).toHaveAttribute("href", "/x");
  });

  it("can carry an accessible caption", () => {
    render(<Table columns={columns} caption="Peer comparison" rows={[]} />);
    expect(screen.getByRole("table", { name: "Peer comparison" })).toBeInTheDocument();
  });
});

describe("ChartFrame", () => {
  const chart = <div data-testid="chart" />;

  it("always renders the chart container, even while loading", () => {
    render(<ChartFrame title="PRICE" state="loading">{chart}</ChartFrame>);
    expect(screen.getByTestId("chart")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveAttribute("aria-busy", "true");
  });

  it("shows an error message as an alert", () => {
    render(<ChartFrame title="PRICE" state="error" errorMessage="Couldn't load price history.">{chart}</ChartFrame>);
    expect(screen.getByRole("alert")).toHaveTextContent("Couldn't load price history.");
  });

  it("shows the empty message", () => {
    render(<ChartFrame title="PRICE" state="empty" emptyMessage="Nothing for this range.">{chart}</ChartFrame>);
    expect(screen.getByText("Nothing for this range.")).toBeInTheDocument();
  });

  it("shows no overlay when ready, and renders title, controls, banner, toolbar and caption", () => {
    render(
      <ChartFrame title="PRICE" controls={<span>RANGE</span>} banner={<span>BANNER</span>} toolbar={<span>TOOLS</span>} caption="Daily bars">
        {chart}
      </ChartFrame>,
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    for (const t of ["PRICE", "RANGE", "BANNER", "TOOLS", "Daily bars"]) expect(screen.getByText(t)).toBeInTheDocument();
  });
});
