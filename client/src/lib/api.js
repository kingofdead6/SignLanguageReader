
const raw = "https://signlanguagereader.onrender.com" 

export const API_BASE = raw
 
export const WS_URLS = (() => {
  if (!raw) {
    return ["ws://127.0.0.1:8000/ws", "ws://localhost:8000/ws"];
  }
  const toWs = (b) => b.replace(/^http/, "ws") + "/ws"; // https -> wss
  const urls = [toWs(API_BASE)];
  if (API_BASE.includes("//localhost")) {
    urls.push(toWs(API_BASE.replace("//localhost", "//127.0.0.1")));
  } else if (API_BASE.includes("//127.0.0.1")) {
    urls.push(toWs(API_BASE.replace("//127.0.0.1", "//localhost")));
  }
  return urls;
})();