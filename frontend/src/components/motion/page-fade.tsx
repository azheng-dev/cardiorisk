"use client";

import { type HTMLMotionProps, motion, useReducedMotion } from "framer-motion";

/**
 * Page-fade wrapper used around the main column of every workflow
 * screen. Cheap (opacity + small Y translate, 220ms ease-out), and
 * collapses to a no-op transition under `prefers-reduced-motion` so
 * vestibular users don't get the swipe.
 *
 * Why Framer Motion and not raw CSS:
 *  - Per-route transitions need an exit phase that survives Next's
 *    route swap; React's `<CSSTransition>` is the wrong abstraction.
 *  - `useReducedMotion()` is one hook call vs a media-query JS dance.
 */
export function PageFade(props: HTMLMotionProps<"div">) {
  const reduced = useReducedMotion();
  if (reduced) {
    return <motion.div {...props} />;
  }
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }}
      {...props}
    />
  );
}
