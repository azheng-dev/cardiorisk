"use client";

import {
  ActivitySquare,
  ClipboardList,
  FileText,
  Gauge,
  History,
  ShieldAlert,
  Stethoscope,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { Logo } from "@/components/brand/logo";
import { ThemeToggle } from "@/components/theme-toggle";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/cn";

/**
 * App shell used by every `/cases/*` route. Sidebar mirrors the
 * 4-stage agent workflow (triage → risk → guideline → letter) plus
 * the audit log so the user always knows where they are in the HITL
 * loop. The "Synthetic data only" banner is persistent across every
 * screen, per AGENTS §1 and ADR-020.
 */

interface NavItem {
  href: string;
  label: string;
  icon: typeof Gauge;
  description: string;
}

const STAGE_NAV: ReadonlyArray<NavItem> = [
  {
    href: "/cases/new",
    label: "New case",
    icon: ClipboardList,
    description: "Patient input form (triage)",
  },
];

function caseNav(caseId: string): ReadonlyArray<NavItem> {
  return [
    {
      href: `/cases/${caseId}/risk`,
      label: "Risk dashboard",
      icon: Gauge,
      description: "Calibrated probability + SHAP attribution",
    },
    {
      href: `/cases/${caseId}/guideline`,
      label: "Guideline",
      icon: ShieldAlert,
      description: "RACGP / NVDPA citations + verifier verdicts",
    },
    {
      href: `/cases/${caseId}/letter`,
      label: "Letter",
      icon: FileText,
      description: "Specialist referral draft (HITL)",
    },
    {
      href: `/cases/${caseId}/audit`,
      label: "Audit log",
      icon: History,
      description: "Per-stage timing, retries, decisions",
    },
  ];
}

export function AppShell({
  caseId,
  children,
}: {
  caseId?: string;
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const items: NavItem[] = [...STAGE_NAV, ...(caseId ? caseNav(caseId) : [])];

  return (
    <div className="min-h-screen bg-[var(--color-bg)] text-[var(--color-fg)]">
      <header className="sticky top-0 z-40 border-b border-[var(--color-border)] bg-[var(--color-surface)]/85 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-6 py-3">
          <div className="flex items-center gap-4">
            <Link href="/" aria-label="CardioRisk Co-Pilot home">
              <Logo size="sm" />
            </Link>
            <Separator orientation="vertical" className="h-6" />
            <Badge variant="warning">
              <ActivitySquare className="size-3.5" aria-hidden /> Synthetic data only
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm">
              <Link href="/brand">Brand</Link>
            </Button>
            <Button asChild variant="ghost" size="sm">
              <a href="https://github.com/azheng-dev/cardiorisk" target="_blank" rel="noreferrer">
                GitHub
              </a>
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-7xl gap-6 px-6 py-6 lg:py-8">
        <aside className="hidden w-60 shrink-0 lg:block" aria-label="Workflow navigation">
          <nav>
            <p className="mb-2 px-2 font-medium text-[var(--color-fg-subtle)] text-xs uppercase tracking-wider">
              Workflow
            </p>
            <ul className="flex flex-col gap-1">
              {items.map((item) => {
                const active =
                  pathname === item.href ||
                  (item.href !== "/cases/new" && pathname.startsWith(item.href));
                const Icon = item.icon;
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      aria-current={active ? "page" : undefined}
                      className={cn(
                        "flex items-start gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                        "hover:bg-[var(--color-surface-muted)]",
                        active
                          ? "bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
                          : "text-[var(--color-fg)]",
                      )}
                    >
                      <Icon className="mt-0.5 size-4 shrink-0" aria-hidden />
                      <span className="flex flex-col">
                        <span className="font-medium leading-tight">{item.label}</span>
                        <span className="text-[var(--color-fg-subtle)] text-xs leading-tight">
                          {item.description}
                        </span>
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ul>
            {!caseId && (
              <p className="mt-6 rounded-md bg-[var(--color-surface-muted)] p-3 text-[var(--color-fg-subtle)] text-xs">
                <Stethoscope className="mb-1 size-3.5 text-[var(--color-fg-muted)]" aria-hidden />{" "}
                Start a new case to unlock the risk, guideline, letter, and audit screens.
              </p>
            )}
          </nav>
        </aside>

        <main className="min-w-0 flex-1">{children}</main>
      </div>
    </div>
  );
}
