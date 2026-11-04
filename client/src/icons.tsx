// Inline SVG icons: 24-unit grid, stroke follows the text colour. All are
// decorative; the owning control carries the accessible name.
import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: string };

function Icon({ size = "1.25rem", children, ...props }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {children}
    </svg>
  );
}

export const ExpandIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M14 4h6v6M20 4l-7 7M10 20H4v-6M4 20l7-7" />
  </Icon>
);

export const ShrinkIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M20 10h-6V4M14 10l7-7M4 14h6v6M10 14l-7 7" />
  </Icon>
);

export const CloseIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Icon>
);

export const SendIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 19V5M5 12l7-7 7 7" />
  </Icon>
);

export const StopIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" stroke="none" />
  </Icon>
);

export const SearchIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="M20 20l-3.5-3.5" />
  </Icon>
);

export const CalendarIcon = (p: IconProps) => (
  <Icon {...p}>
    <rect x="3" y="5" width="18" height="16" rx="2" />
    <path d="M3 10h18M8 3v4M16 3v4" />
  </Icon>
);

export const PaperPlaneIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M21 3L10 14M21 3l-7 18-4-7-7-4z" />
  </Icon>
);

export const CogIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="3" />
    <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1" />
  </Icon>
);

export const CheckIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5 12l5 5L20 7" />
  </Icon>
);

export const ClockIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </Icon>
);

export const AlertIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3l10 18H2L12 3zM12 10v4M12 17.5v.5" />
  </Icon>
);

export const ThumbUpIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M7 11v9M3 11h4l4-8a2.5 2.5 0 0 1 2.5 2.5V10h4.6a2 2 0 0 1 2 2.3l-1.1 6A2 2 0 0 1 17 20H7" />
  </Icon>
);

export const ThumbDownIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M17 13V4M21 13h-4l-4 8a2.5 2.5 0 0 1-2.5-2.5V14H5.9a2 2 0 0 1-2-2.3l1.1-6A2 2 0 0 1 7 4h10" />
  </Icon>
);

export const ArrowDownIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 5v14M5 12l7 7 7-7" />
  </Icon>
);
