mixins.home = {
    mounted() {
        let background = this.$refs.homeBackground;
        let images = background.dataset.images.split(",");
        let id = Math.floor(Math.random() * images.length);
        background.style.backgroundImage = `url('${images[id]}')`;
        this.menuColor = true;

        // Apply blur effect based on scroll
        const homePostsWrap = this.$refs.homePostsWrap;

        const maxBlur = 10; // Max blur: 10px
        const startBlurOffset = 0.618;
        if (homePostsWrap && background) {
            window.addEventListener("scroll", () => {
                const rect = homePostsWrap.getBoundingClientRect();
                const windowHeight = window.innerHeight * startBlurOffset; // Height of the viewport
                offsetFromTop = 0;
                if (rect.top < windowHeight) {
                    offsetFromTop = Math.floor((windowHeight - rect.top) / windowHeight * maxBlur); // How much it's scrolled past the top of the viewport}
                }

                // Blur increases as homePostsWrap reaches the top
                let blurValue = Math.min(offsetFromTop, maxBlur);
                background.style.filter = `blur(${blurValue}px)`;
            });
        }
    },
    methods: {
        homeClick() {
            window.scrollTo({ top: window.innerHeight, behavior: "smooth" });
        },
    },
};