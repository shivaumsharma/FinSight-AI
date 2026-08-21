// getUserMedia()'s returned promise only settles once the user answers
// the browser's own permission prompt (or the OS/browser auto-denies
// it) -- there is no built-in timeout for a prompt that's never
// answered (missed, dismissed by clicking elsewhere, or just never
// noticed). Found live in VoiceInputButton.tsx's own voice session:
// this left the UI stuck on a default "starting" label indefinitely,
// a class of stuck-forever bug none of that component's other
// watchdogs cover, since its MAX_RECORDING_MS timer only starts
// counting once recording actually begins -- which never happens if
// getUserMedia() itself never resolves. Shared here (not defined once
// per caller) since useWakeWordListener.ts's start() has the exact
// same unprotected call and the exact same failure mode.
const GET_USER_MEDIA_TIMEOUT_MS = 20_000;

export class GetUserMediaTimeoutError extends Error {}

export function getUserMediaWithTimeout(): Promise<MediaStream> {
  let didTimeOut = false;
  const streamPromise = navigator.mediaDevices.getUserMedia({ audio: true });
  // If the real prompt eventually gets answered well after this
  // timeout already gave up (and moved the caller to an error state),
  // the resulting stream would otherwise leak -- nothing else holds a
  // reference to stop it. Release it immediately instead of leaving
  // the mic silently held open.
  streamPromise.then(
    (stream) => { if (didTimeOut) stream.getTracks().forEach((t) => t.stop()); },
    () => {}
  );
  return Promise.race([
    streamPromise,
    new Promise<MediaStream>((_, reject) =>
      window.setTimeout(() => {
        didTimeOut = true;
        reject(new GetUserMediaTimeoutError());
      }, GET_USER_MEDIA_TIMEOUT_MS)
    ),
  ]);
}
