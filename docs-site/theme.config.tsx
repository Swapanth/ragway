import { DocsThemeConfig } from 'nextra-theme-docs'
import { useRouter } from 'next/router'

const config: DocsThemeConfig = {
    logo: <span style={{ fontFamily: 'DM Mono, monospace', fontWeight: 500 }}>ragway</span>,
    project: {
        link: 'https://github.com/yourusername/ragway',
    },
    docsRepositoryBase: 'https://github.com/yourusername/ragway/tree/main/docs-site',
    navigation: true,
    darkMode: true,
    nextThemes: { defaultTheme: 'dark' },
    footer: {
        text: (
            <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 12 }}>
                ragway · The way to build RAG · MIT License
            </span>
        ),
    },
    primaryHue: 200,
    primarySaturation: 88,
    useNextSeoProps() {
        return {
            titleTemplate: '%s · ragway',
            description: 'The way to build RAG — modular, configurable, plugin-based.',
            openGraph: {
                images: [{ url: '/og.png' }],
            },
        }
    },
    head: function Head() {
        const router = useRouter()
        const canonical = `https://ragway.dev${router.asPath === '/' ? '' : router.asPath}`

        return (
            <>
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                <meta name="theme-color" content="#0a0a0a" />
                <meta property="og:title" content="ragway docs" />
                <link rel="canonical" href={canonical} />
            </>
        )
    },
    sidebar: {
        defaultMenuCollapseLevel: 1,
        toggleButton: true,
    },
    toc: { float: true },
    editLink: { text: 'Edit this page on GitHub →' },
    feedback: { content: null },
    banner: {
        key: 'v0.1.0',
        text: (
            <span>
                ragway v0.1.0 is live on PyPI →{' '}
                <code
                    style={{
                        background: '#0b0b0b',
                        border: '1px solid #1f1f1f',
                        borderRadius: 4,
                        padding: '0.1rem 0.4rem',
                        fontFamily: 'DM Mono, monospace',
                    }}
                >
                    pip install ragway
                </code>
            </span>
        ),
    },
}

export default config
