"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { ArrowRight, Loader2, Sparkles, Stethoscope } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import type { z } from "zod";

import { AppShell } from "@/components/app-shell/app-shell";
import { ErrorState } from "@/components/domain/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { SAMPLE_PATIENT, isMockMode } from "@/lib/agents/client";
import { patientInputSchema } from "@/lib/agents/schema";
import { useCaseStore } from "@/lib/agents/store";

const formSchema = patientInputSchema;
type FormValues = z.infer<typeof formSchema>;

const DEFAULTS: FormValues = {
  Age: 55,
  Sex: "M",
  ChestPainType: "ASY",
  RestingBP: 130,
  Cholesterol: 220,
  FastingBS: 0,
  RestingECG: "Normal",
  MaxHR: 140,
  ExerciseAngina: "N",
  Oldpeak: 1.0,
  ST_Slope: "Up",
};

export default function NewCasePage() {
  const router = useRouter();
  const { start, loading, error, reset } = useCaseStore();
  const [submitting, setSubmitting] = useState(false);

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: DEFAULTS,
    mode: "onBlur",
  });

  async function onSubmit(values: FormValues) {
    setSubmitting(true);
    try {
      const snap = await start(values);
      router.push(`/cases/${snap.case_id}/risk`);
    } catch {
      setSubmitting(false);
    }
  }

  function loadSample() {
    form.reset(SAMPLE_PATIENT);
  }

  function clearForm() {
    reset();
    form.reset(DEFAULTS);
  }

  return (
    <AppShell>
      <div className="flex flex-col gap-6">
        <header className="flex flex-col gap-3">
          <Badge variant="accent">
            <Stethoscope className="size-3.5" aria-hidden /> Stage 1 — Triage
          </Badge>
          <h1 className="font-display font-semibold text-3xl tracking-tight sm:text-4xl">
            New patient case
          </h1>
          <p className="max-w-3xl text-[var(--color-fg-muted)]">
            Submit synthetic vitals and history. The triage agent normalises the input, then hands
            off to the calibrated risk model. Every field maps 1:1 to the Heart Failure Prediction
            (HFP) schema the v1 model expects.
          </p>
        </header>

        {isMockMode() && (
          <div className="rounded-md border border-[var(--color-info)]/30 bg-[var(--color-info-soft)] px-4 py-3 text-[var(--color-info)] text-sm">
            <strong>Mock mode is on</strong> (`NEXT_PUBLIC_AGENT_MOCK=true`). Submissions return a
            deterministic sample case so the UI renders without a backend. Switch to live by
            unsetting the env var and pointing `NEXT_PUBLIC_API_BASE_URL` at the FastAPI surface.
          </div>
        )}

        {error && (
          <ErrorState
            title="Could not start case"
            description={error}
            action={
              <Button variant="outline" size="sm" onClick={clearForm}>
                Reset form
              </Button>
            }
          />
        )}

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="flex flex-col gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Demographics</CardTitle>
                <CardDescription>Age and sex at birth.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-6 sm:grid-cols-2">
                <FormField
                  control={form.control}
                  name="Age"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Age (years)</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          min={18}
                          max={120}
                          {...field}
                          onChange={(e) => field.onChange(Number(e.target.value))}
                        />
                      </FormControl>
                      <FormDescription>18 – 120.</FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="Sex"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Sex at birth</FormLabel>
                      <FormControl>
                        <Select value={field.value} onValueChange={field.onChange}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="M">Male</SelectItem>
                            <SelectItem value="F">Female</SelectItem>
                          </SelectContent>
                        </Select>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Vitals</CardTitle>
                <CardDescription>Resting clinic measurements at the index visit.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                <FormField
                  control={form.control}
                  name="RestingBP"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Resting BP (mmHg)</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          min={60}
                          max={260}
                          {...field}
                          onChange={(e) => field.onChange(Number(e.target.value))}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="MaxHR"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Max HR (bpm)</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          min={50}
                          max={240}
                          {...field}
                          onChange={(e) => field.onChange(Number(e.target.value))}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="Cholesterol"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Total cholesterol (mg/dL)</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          min={0}
                          max={800}
                          {...field}
                          onChange={(e) => field.onChange(Number(e.target.value))}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="Oldpeak"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Oldpeak (ST depression)</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          step={0.1}
                          min={-3}
                          max={8}
                          {...field}
                          onChange={(e) => field.onChange(Number(e.target.value))}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="FastingBS"
                  render={({ field }) => (
                    <FormItem className="sm:col-span-2 lg:col-span-1">
                      <FormLabel>Fasting blood sugar &gt; 120 mg/dL</FormLabel>
                      <FormControl>
                        <div className="flex items-center gap-3">
                          <Switch
                            checked={field.value === 1}
                            onCheckedChange={(v) => field.onChange(v ? 1 : 0)}
                          />
                          <span className="text-[var(--color-fg-muted)] text-sm">
                            {field.value === 1 ? "Hyperglycaemic" : "Normoglycaemic"}
                          </span>
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>History &amp; ECG</CardTitle>
                <CardDescription>
                  Symptom pattern and resting electrocardiogram findings.
                </CardDescription>
              </CardHeader>
              <CardContent className="grid gap-6 sm:grid-cols-2">
                <FormField
                  control={form.control}
                  name="ChestPainType"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Chest-pain type</FormLabel>
                      <FormControl>
                        <Select value={field.value} onValueChange={field.onChange}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="TA">Typical angina (TA)</SelectItem>
                            <SelectItem value="ATA">Atypical angina (ATA)</SelectItem>
                            <SelectItem value="NAP">Non-anginal pain (NAP)</SelectItem>
                            <SelectItem value="ASY">Asymptomatic (ASY)</SelectItem>
                          </SelectContent>
                        </Select>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="RestingECG"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Resting ECG</FormLabel>
                      <FormControl>
                        <Select value={field.value} onValueChange={field.onChange}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="Normal">Normal</SelectItem>
                            <SelectItem value="ST">ST-T wave abnormality</SelectItem>
                            <SelectItem value="LVH">Left-ventricular hypertrophy</SelectItem>
                          </SelectContent>
                        </Select>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="ExerciseAngina"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Exercise-induced angina</FormLabel>
                      <FormControl>
                        <RadioGroup
                          value={field.value}
                          onValueChange={field.onChange}
                          className="flex gap-4"
                        >
                          <div className="flex items-center gap-2">
                            <RadioGroupItem value="Y" id="angina-y" />
                            <Label htmlFor="angina-y" className="font-normal">
                              Yes
                            </Label>
                          </div>
                          <div className="flex items-center gap-2">
                            <RadioGroupItem value="N" id="angina-n" />
                            <Label htmlFor="angina-n" className="font-normal">
                              No
                            </Label>
                          </div>
                        </RadioGroup>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="ST_Slope"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>ST-segment slope (peak exercise)</FormLabel>
                      <FormControl>
                        <Select value={field.value} onValueChange={field.onChange}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="Up">Up</SelectItem>
                            <SelectItem value="Flat">Flat</SelectItem>
                            <SelectItem value="Down">Down</SelectItem>
                          </SelectContent>
                        </Select>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </CardContent>
              <CardFooter className="flex flex-wrap items-center justify-between gap-3">
                <Button type="button" variant="ghost" size="sm" onClick={loadSample}>
                  <Sparkles className="size-4" aria-hidden /> Load sample patient
                </Button>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={clearForm}
                    disabled={submitting || loading}
                  >
                    Reset
                  </Button>
                  <Button type="submit" size="lg" disabled={submitting || loading}>
                    {submitting || loading ? (
                      <>
                        <Loader2
                          className="size-4 animate-spin motion-reduce:animate-none"
                          aria-hidden
                        />
                        Triaging…
                      </>
                    ) : (
                      <>
                        Triage &amp; score risk
                        <ArrowRight className="size-4" aria-hidden />
                      </>
                    )}
                  </Button>
                </div>
              </CardFooter>
            </Card>
          </form>
        </Form>
      </div>
    </AppShell>
  );
}
