// modified from butterfly

document.addEventListener('DOMContentLoaded', () => {
    /**
    * toc,anchor
    */
    const newTocNode = (level, idStack, item, index) => {
        const li = document.createElement('li')
        li.className = 'toc-item toc-level-' + level
        const link_span = document.createElement('span')
        link_span.className = 'toc-link'
        // link_span.href = '#' + encodeURI(item.id)
        link_span.id = `toc-link-${item.id}`
        // add attibute "winTop" to link_span
        // link_span.setAttribute('winTop', item.top)
        link_span.addEventListener('click', (e) => {
            e.preventDefault();
            scrollToDest(item.top - offsetY + 80);
        }
        )
        const spanNumber = document.createElement('span')
        spanNumber.className = 'toc-number'
        spanNumber.textContent = idStack.join('.') + '.'
        link_span.appendChild(spanNumber)
        const spanText = document.createElement('span')
        spanText.className = 'toc-text'
        spanText.textContent = item.textContent
        link_span.appendChild(spanText)
        li.appendChild(link_span)
        return li
    }

    const buildToc = (l) => {
        const validList = l.map(item => item.level)
        const minId = Math.min(...validList)
        const idOffset = minId - 1;

        let currentNode = document.createElement('ol');
        currentNode.classList.add('toc');
        const nodeList = [];
        const idStack = [0];
        l.forEach((item, index) => {
            level = item.level - idOffset

            while (level < idStack.length) {
                nodeList[nodeList.length - 1].appendChild(currentNode)
                currentNode = nodeList.pop()
                idStack.pop()
            }
            if (level === idStack.length) {
                idStack[idStack.length - 1] += 1;
            }
            else { // level > idStack.length
                nodeList.push(currentNode);
                currentNode = document.createElement('ol');
                currentNode.classList.add('toc-child');
                idStack.push(1);
            }
            currentNode.appendChild(newTocNode(level, idStack, item, index));
        });

        while (idStack.length > 1) {
            nodeList[nodeList.length - 1].appendChild(currentNode)
            currentNode = nodeList.pop()
            idStack.pop()
        }

        const tocContent = document.getElementById("toc-content")
        tocContent.appendChild(currentNode)
    }

    const scrollToDest = (pos, time = 500) => {
        const currentPos = window.scrollY
        const isNavFixed = !document.getElementById('menu').classList.contains('hidden')
        if (currentPos > pos || isNavFixed) pos = pos - 50

        if ('scrollBehavior' in document.documentElement.style) {
            window.scrollTo({
                top: pos,
                behavior: 'smooth'
            })
            return
        }

        const startTime = performance.now()
        const animate = currentTime => {
            const timeElapsed = currentTime - startTime
            const progress = Math.min(timeElapsed / time, 1)
            window.scrollTo(0, currentPos + (pos - currentPos) * progress)
            if (progress < 1) {
                requestAnimationFrame(animate)
            }
        }
        requestAnimationFrame(animate)
    }

    const getEleTop = (ele) => {
        if (!ele) return 0

        let actualTop = ele.offsetTop
        let current = ele.offsetParent

        while (current !== null) {
            actualTop += current.offsetTop
            current = current.offsetParent
        }

        return actualTop
    }

    const updateAnchor = (anchor) => {
        if (anchor !== window.location.hash) {
            if (!anchor) anchor = location.pathname
            const title = document.title
            window.history.replaceState({
                url: location.href,
                title
            }, title, anchor)
        }
    }

    const findHeadPosition = (top) => {

        let currentId = encodeURI($articleList[0].id)

        let currentIndex = 0

        if (top > 0) {
            for (let i = 0; i < $articleList.length; i++) {
                const ele = $articleList[i]

                const topOffset = ele.top;

                if (top > topOffset) {
                    currentId = ele.id ? '#' + encodeURI(ele.id) : ''
                    currentIndex = i
                } else {
                    break
                }
            }
        }
        // if (detectItem === currentIndex) return

        if (isAnchor) updateAnchor(currentId)

        detectItem = currentIndex
        $tocLink = $cardToc.querySelectorAll('.toc-link')

        if (isToc) {
            console.log(detectItem)

            document.getElementById("toc-content").querySelectorAll('.toc-active').forEach(i => i.classList.remove('toc-active'))

            if (currentId) {
                const currentActive = document.getElementById($tocLink[currentIndex].id)
                currentActive.classList.add('toc-active')

                // scroll to active toc in aside bar
                // setTimeout(() => autoScrollToc(currentActive), 0)

                if (!isExpand) {
                    let parent = currentActive.parentNode
                    while (!parent.matches('.toc')) {
                        if (parent.matches('li')) parent.classList.add('toc-active')
                        parent = parent.parentNode
                    }
                }
            }
        }
    }


    const throttle = (func, wait, options = {}) => {
        let timeout, context, args;
        let previous = 0;

        const later = () => {
            previous = options.leading === false ? 0 : new Date().getTime();
            timeout = null;
            func.apply(context, args);
            if (!timeout) context = args = null;
        };

        const throttled = function (...params) {
            const now = new Date().getTime();
            if (!previous && options.leading === false) previous = now;
            const remaining = wait - (now - previous);
            context = this;
            args = params;
            if (remaining <= 0 || remaining > wait) {
                if (timeout) {
                    clearTimeout(timeout);
                    timeout = null;
                }
                previous = now;
                func.apply(context, args);
                if (!timeout) context = args = null;
            } else if (!timeout && options.trailing !== false) {
                timeout = setTimeout(later, remaining);
            }
        };

        return throttled;
    }

    const getScrollPercent = (currentTop, ele) => {
        if (!docHeight || ele.clientHeight !== docHeight) {
            docHeight = ele.clientHeight;
            winHeight = window.innerHeight;
            headerHeight = ele.offsetTop;
            contentMath = Math.max(docHeight - winHeight, document.documentElement.scrollHeight - winHeight);
        }

        const scrollPercent = (currentTop - headerHeight) / contentMath;
        return `${Math.max(0, Math.min(100, Math.round(scrollPercent * 100)))}%`;
    }

    const addEventListener = (ele, event, fn, option = false) => {

        ele.addEventListener(event, fn, option);

    }


    const autoScrollToc = item => {
        const sidebarHeight = $cardToc.clientHeight
        const itemOffsetTop = item.offsetTop
        const itemHeight = item.clientHeight
        const scrollTop = $cardToc.scrollTop
        const offset = itemOffsetTop - scrollTop
        const middlePosition = (sidebarHeight - itemHeight) / 2

        if (offset !== middlePosition) {
            $cardToc.scrollTop = scrollTop + (offset - middlePosition)
        }
    }

    // main of scroll
    const tocScrollFn = throttle(() => {
        const currentTop = window.scrollY || document.documentElement.scrollTop
        document.getElementById('toc-percentage').textContent = getScrollPercent(currentTop, $article)
        findHeadPosition(currentTop + offsetY)
    }, 100)

    let docHeight, winHeight, headerHeight, contentMath
    let $tocLink, isExpand

    const $article = document.getElementById('content')
    const offsetY = window.innerHeight * 0.382
    // const offsetY = window.innerHeight / 2
    const $articleList = Array.from(
        $article.querySelectorAll('h1,h2,h3,h4,h5,h6')
    ).map(
        item => ({
            tag: item.tagName,
            id: item.id,
            top: getEleTop(item),
            textContent: item.textContent,
            level: Number(item.tagName.slice(1)),
        })
    )

    buildToc($articleList)

    let detectItem = 0

    // const $cardTocLayout = document.getElementById('card-toc')
    const $cardToc = document.getElementById("toc-content")
    isExpand = $cardToc.classList.contains('is-expand')

    addEventListener(window, 'scroll', tocScrollFn, { passive: true })

});
