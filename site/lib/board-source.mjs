/** Prefer the newest valid publication; CDN endpoints can update at different times. */
export function newestBoard(results) {
  const valid = results.filter(b => b && Number.isFinite(Date.parse(b.generated_at)) &&
    typeof b.status === 'string' && Array.isArray(b.recommendations) && Array.isArray(b.watchlist));
  if (!valid.length) throw new Error('No valid model board available');
  return valid.reduce((latest, board) => Date.parse(board.generated_at) > Date.parse(latest.generated_at) ? board : latest);
}
export async function fetchLatestBoard(fetcher = fetch) {
  const stamp = Date.now();
  const urls = [
    'https://drhyphy.github.io/nfl-qb-attempts/board.json',
    'https://raw.githubusercontent.com/drhyphy/nfl-qb-attempts/main/data/published/latest.json',
    './board.json',
  ];
  const results = await Promise.allSettled(urls.map(async url => {
    const response = await fetcher(`${url}?t=${stamp}`, {cache:'no-store', signal:AbortSignal.timeout(12000)});
    if (!response.ok) throw new Error('Board source unavailable');
    return response.json();
  }));
  return newestBoard(results.filter(r => r.status === 'fulfilled').map(r => r.value));
}
