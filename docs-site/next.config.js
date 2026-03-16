const withNextra = require('nextra').default({
    theme: 'nextra-theme-docs',
    themeConfig: './theme.config.tsx',
    defaultShowCopyCode: true,
})

module.exports = withNextra({
    output: 'export',
    images: { unoptimized: true },
    trailingSlash: true,
})
