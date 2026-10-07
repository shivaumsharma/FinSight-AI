"use client";

// Last-resort boundary for errors thrown in the root layout itself. It
// replaces the layout, so it must bring its own <html>/<body> and cannot rely
// on globals.css or the theme tokens -- hence the inline styles.
export default function GlobalError({
  error,
  unstable_retry,
}: {
  error: Error & { digest?: string };
  unstable_retry: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#05070a",
          color: "#e6e9ee",
          fontFamily: "ui-monospace, Menlo, Consolas, monospace",
          padding: 20,
          textAlign: "center",
        }}
      >
        <div role="alert" style={{ maxWidth: 360 }} data-digest={error.digest}>
          <p style={{ fontSize: 14 }}>Something went wrong.</p>
          <button
            type="button"
            onClick={() => unstable_retry()}
            style={{
              marginTop: 16,
              padding: "8px 16px",
              background: "#0c1118",
              color: "#e6e9ee",
              border: "1px solid #2a3340",
              borderRadius: 8,
              fontFamily: "inherit",
              fontSize: 12,
              fontWeight: 700,
              cursor: "pointer",
            }}
          >
            TRY AGAIN
          </button>
        </div>
      </body>
    </html>
  );
}
