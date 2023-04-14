import '@/styles/base.css';
import { Inter } from 'next/font/google';
const inter = Inter({
    variable: '--font-inter',
    subsets: ['latin'],
});
function MyApp({ Component, pageProps }) {
    return (<>
      <main className={inter.variable}>
        <Component {...pageProps}/>
      </main>
    </>);
}
export default MyApp;
