import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useWakeWordListener } from "./useWakeWordListener";

// See useRealtimeVoiceInput.test.ts's own header -- same class of bug,
// same fix shape: this hook had no useEffect cleanup at all before this
// change, so a user navigating away while the always-on wake-word
// listener was running left the mic and WebSocket open indefinitely.

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = 0;
  closed = false;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }
  send() {}
  close() {
    this.closed = true;
    this.readyState = 3;
  }
}

function installFakes() {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);

  const stopTrack = vi.fn();
  const fakeStream = { getTracks: () => [{ stop: stopTrack }] };
  vi.stubGlobal("navigator", {
    mediaDevices: { getUserMedia: vi.fn().mockResolvedValue(fakeStream) },
  });

  const disconnect = vi.fn();
  const close = vi.fn().mockResolvedValue(undefined);
  class FakeAudioContext {
    destination = {};
    createMediaStreamSource() {
      return { connect: vi.fn(), disconnect };
    }
    createScriptProcessor() {
      return { onaudioprocess: null, connect: vi.fn(), disconnect };
    }
    createGain() {
      return { gain: { value: 0 }, connect: vi.fn(), disconnect };
    }
    close = close;
  }
  vi.stubGlobal("AudioContext", FakeAudioContext);

  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ token: "tok", ws_url: "wss://example.com/ws" }),
    })
  );

  return { stopTrack, close };
}

describe("useWakeWordListener", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("releases the mic, AudioContext, and WebSocket on unmount while running", async () => {
    const { stopTrack, close } = installFakes();
    const { result, unmount } = renderHook(() => useWakeWordListener(vi.fn()));

    await act(async () => {
      await result.current.start();
    });

    expect(FakeWebSocket.instances).toHaveLength(1);
    const ws = FakeWebSocket.instances[0];
    expect(ws.closed).toBe(false);
    expect(stopTrack).not.toHaveBeenCalled();

    unmount();

    expect(stopTrack).toHaveBeenCalledTimes(1);
    expect(close).toHaveBeenCalledTimes(1);
    expect(ws.closed).toBe(true);
  });

  it("unmounting before start() was ever called does nothing and does not throw", () => {
    const { stopTrack, close } = installFakes();
    const { unmount } = renderHook(() => useWakeWordListener(vi.fn()));

    expect(() => unmount()).not.toThrow();
    expect(stopTrack).not.toHaveBeenCalled();
    expect(close).not.toHaveBeenCalled();
  });

  it("unmounting after an explicit stop() does not double-release", async () => {
    const { stopTrack, close } = installFakes();
    const { result, unmount } = renderHook(() => useWakeWordListener(vi.fn()));

    await act(async () => {
      await result.current.start();
    });
    act(() => {
      result.current.stop();
    });

    stopTrack.mockClear();
    close.mockClear();

    unmount();

    expect(stopTrack).not.toHaveBeenCalled();
    expect(close).not.toHaveBeenCalled();
  });
});
