import type {Metadata} from 'next';
import './globals.css';
export const metadata:Metadata={title:'Attempts | NFL QB Props',description:'Daily NFL quarterback pass-attempt projections and market comparisons. Updated each morning at 6:30 Eastern.'};
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}
