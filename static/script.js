/* ==========================================================================
   Crop Yield Predictor - frontend logic
   1. loads dropdown values from   GET  /api/options
   2. validates the form in the browser
   3. sends the values to          POST /predict   (fetch, no page reload)
   4. shows the result card
   ========================================================================== */

(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const form = $("#predict-form");
  const banner = $("#banner");
  const predictBtn = $("#predict-btn");

  const state = { options: null, activeExample: null };

  // nf(2)(1234.5678) -> "1,234.57"
  const nf = (digits) => {
    const formatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: digits });
    return (value) => formatter.format(value);
  };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ------------------------------------------------------------------ */
  /* Field helpers                                                      */
  /* ------------------------------------------------------------------ */
  function fieldBox(name) { return $(`.field[data-field="${name}"]`); }

  function setError(name, message) {
    const box = fieldBox(name);
    if (!box) return;
    box.classList.add("invalid");
    $("[data-error]", box).textContent = message;
    $("input", box).setAttribute("aria-invalid", "true");
  }

  function clearError(name) {
    const box = fieldBox(name);
    if (!box) return;
    box.classList.remove("invalid");
    $("[data-error]", box).textContent = "";
    $("input", box).removeAttribute("aria-invalid");
  }

  function clearAllErrors() {
    ["country", "crop", "year", "rainfall", "pesticides", "temperature"].forEach(clearError);
    banner.hidden = true;
  }

  function showBanner(message) {
    banner.textContent = message;
    banner.hidden = false;
  }

  /* ------------------------------------------------------------------ */
  /* Searchable dropdown (combobox)                                     */
  /* ------------------------------------------------------------------ */
  function makeCombobox(input, list, getItems) {
    let active = -1;
    let shown = [];
    let silent = false; // true while choose() fires its own input event

    function highlight(text, query) {
      const frag = document.createDocumentFragment();
      const at = query ? text.toLowerCase().indexOf(query) : -1;
      if (at < 0) { frag.append(text); return frag; }
      frag.append(text.slice(0, at));
      const mark = document.createElement("mark");
      mark.textContent = text.slice(at, at + query.length);
      frag.append(mark, text.slice(at + query.length));
      return frag;
    }

    function render() {
      const query = input.value.trim().toLowerCase();
      shown = getItems().filter((item) => item.toLowerCase().includes(query));
      list.replaceChildren();
      if (!shown.length) {
        const none = document.createElement("li");
        none.className = "none";
        none.textContent = "No match in the dataset";
        list.append(none);
      }
      shown.forEach((item, index) => {
        const li = document.createElement("li");
        li.setAttribute("role", "option");
        li.setAttribute("aria-selected", index === active ? "true" : "false");
        li.append(highlight(item, query));
        li.addEventListener("mousedown", (event) => { event.preventDefault(); choose(item); });
        list.append(li);
      });
    }

    function open() {
      active = -1;
      render();
      list.hidden = false;
      input.setAttribute("aria-expanded", "true");
    }

    function close() {
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
    }

    function choose(item) {
      input.value = item;
      close();
      clearError(input.id);
      silent = true;
      input.dispatchEvent(new Event("input", { bubbles: true })); // lets the form notice the change
      silent = false;
    }

    // open on click / typing / arrow keys (not on focus, so code-driven focus never covers other fields)
    input.addEventListener("click", () => { if (list.hidden) open(); });
    input.addEventListener("input", () => {
      if (silent) return;
      if (list.hidden) open(); else { active = -1; render(); }
    });
    input.addEventListener("blur", () => {
      close();
      // if the text matches an item ignoring case, use the exact spelling
      const exact = getItems().find((item) => item.toLowerCase() === input.value.trim().toLowerCase());
      if (exact) input.value = exact;
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (list.hidden) open();
        if (!shown.length) return;
        active = (active + (event.key === "ArrowDown" ? 1 : -1) + shown.length) % shown.length;
        render();
        const current = list.children[active];
        if (current) current.scrollIntoView({ block: "nearest" });
      } else if (event.key === "Enter" && !list.hidden && active >= 0) {
        event.preventDefault();
        choose(shown[active]);
      } else if (event.key === "Escape") {
        close();
      }
    });
  }

  /* ------------------------------------------------------------------ */
  /* Load options from the server                                       */
  /* ------------------------------------------------------------------ */
  async function loadOptions() {
    try {
      const response = await fetch("/api/options");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "The model is not available.");
      state.options = data;
    } catch (error) {
      showBanner(error.message || "Could not load the model information. Is the server running?");
      predictBtn.disabled = true;
      return;
    }

    const { countries, crops, ranges, examples, dataset } = state.options;
    makeCombobox($("#country"), $("#country-list"), () => countries);
    makeCombobox($("#crop"), $("#crop-list"), () => crops);

    const year = $("#year");
    year.min = ranges.year.min;
    year.max = ranges.year.max;
    $('[data-field="year"] [data-hint]').textContent = `Dataset years ${ranges.year.min}–${ranges.year.max}`;
    $('[data-field="temperature"] [data-hint]').textContent = `Dataset range ${nf(1)(ranges.temperature.min)} to ${nf(1)(ranges.temperature.max)} °C`;
    $('[data-field="rainfall"] [data-hint]').textContent = `Dataset range ${nf(0)(ranges.rainfall.min)} to ${nf(0)(ranges.rainfall.max)} mm`;
    $('[data-field="pesticides"] [data-hint]').textContent = `Dataset range ${nf(2)(ranges.pesticides.min)} to ${nf(0)(ranges.pesticides.max)} tonnes`;

    $("#dataset-note").textContent =
      `Trained on ${nf(0)(dataset.rows)} historical records from ${dataset.countries} countries, covering ${dataset.crops} crops between ${ranges.year.min} and ${ranges.year.max}.`;

    if (examples && examples.length) {
      const holder = $("#example-chips");
      examples.slice(0, 4).forEach((example, index) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = `${example.country}, ${example.crop}, ${example.year}`;
        chip.setAttribute("aria-pressed", "false");
        chip.addEventListener("click", () => fillExample(example, chip));
        holder.append(chip);
      });
      $("#examples").hidden = false;
    }
  }

  function fillExample(example, chip) {
    clearAllErrors();
    ["country", "crop", "year", "rainfall", "pesticides", "temperature"].forEach((key) => {
      $("#" + key).value = example[key];
    });
    state.activeExample = example;
    document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", c === chip ? "true" : "false"));
  }

  // Editing any field after loading an example means it is no longer that record
  form.addEventListener("input", (event) => {
    if (event.target.id) clearError(event.target.id);
    if (state.activeExample && !matchesExample(readValues(), state.activeExample)) {
      state.activeExample = null;
      document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
    }
  });

  function matchesExample(values, example) {
    return values.country === example.country && values.crop === example.crop &&
      Number(values.year) === example.year && Number(values.rainfall) === example.rainfall &&
      Number(values.pesticides) === example.pesticides && Number(values.temperature) === example.temperature;
  }

  /* ------------------------------------------------------------------ */
  /* Validation                                                         */
  /* ------------------------------------------------------------------ */
  function readValues() {
    return {
      country: $("#country").value.trim(),
      crop: $("#crop").value.trim(),
      year: $("#year").value.trim(),
      rainfall: $("#rainfall").value.trim(),
      pesticides: $("#pesticides").value.trim(),
      temperature: $("#temperature").value.trim(),
    };
  }

  function validate(values) {
    const problems = {};
    const { countries, crops, ranges } = state.options;

    const country = countries.find((c) => c.toLowerCase() === values.country.toLowerCase());
    if (!values.country) problems.country = "Choose a country.";
    else if (!country) problems.country = "Pick a country from the list.";

    const crop = crops.find((c) => c.toLowerCase() === values.crop.toLowerCase());
    if (!values.crop) problems.crop = "Choose a crop.";
    else if (!crop) problems.crop = "Pick a crop from the list.";

    const num = (key) => (values[key] === "" ? NaN : Number(values[key]));

    const year = num("year");
    if (values.year === "") problems.year = "Enter a year.";
    else if (!Number.isInteger(year) || year < ranges.year.min || year > ranges.year.max)
      problems.year = `Enter a whole year from ${ranges.year.min} to ${ranges.year.max}.`;

    const rain = num("rainfall");
    if (values.rainfall === "") problems.rainfall = "Enter the yearly rainfall.";
    else if (Number.isNaN(rain)) problems.rainfall = "Rainfall must be a number.";
    else if (rain < 0) problems.rainfall = "Rainfall cannot be negative.";
    else if (rain > 12000) problems.rainfall = "Rainfall must be 12,000 mm/year or less.";

    const pest = num("pesticides");
    if (values.pesticides === "") problems.pesticides = "Enter the pesticide usage.";
    else if (Number.isNaN(pest)) problems.pesticides = "Pesticide usage must be a number.";
    else if (pest < 0) problems.pesticides = "Pesticide usage cannot be negative.";

    const temp = num("temperature");
    if (values.temperature === "") problems.temperature = "Enter the average temperature.";
    else if (Number.isNaN(temp)) problems.temperature = "Temperature must be a number.";
    else if (temp < -50 || temp > 60) problems.temperature = "Temperature must be between -50 and 60 °C.";

    return { problems, country, crop };
  }

  /* ------------------------------------------------------------------ */
  /* Submit                                                             */
  /* ------------------------------------------------------------------ */
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.options) return;
    clearAllErrors();

    const values = readValues();
    const { problems, country, crop } = validate(values);
    const names = Object.keys(problems);
    if (names.length) {
      names.forEach((name) => setError(name, problems[name]));
      $("#" + names[0]).focus();
      return;
    }

    const payload = {
      country, crop,
      year: Number(values.year),
      rainfall: Number(values.rainfall),
      pesticides: Number(values.pesticides),
      temperature: Number(values.temperature),
    };

    setLoading(true);
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      let data = {};
      try { data = await response.json(); } catch (_) { /* not JSON */ }

      if (!response.ok) {
        if (data.field) { setError(data.field, data.error); $("#" + data.field).focus(); }
        else showBanner(data.error || "The prediction could not be completed.");
        return;
      }
      showResult(data, payload);
    } catch (error) {
      showBanner("Could not reach the server. Check that python app.py is still running, then try again.");
    } finally {
      setLoading(false);
    }
  });

  function setLoading(isLoading) {
    predictBtn.disabled = isLoading;
    predictBtn.classList.toggle("loading", isLoading);
    $(".btn-label", predictBtn).textContent = isLoading ? "Predicting" : "Predict Crop Yield";
  }

  $("#clear-btn").addEventListener("click", () => {
    form.reset();
    clearAllErrors();
    state.activeExample = null;
    document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
    $("#result").hidden = true;
    $("#empty-state").hidden = false;
    $("#country").focus();
  });

  /* ------------------------------------------------------------------ */
  /* Result card                                                        */
  /* ------------------------------------------------------------------ */
  function showResult(data, sent) {
    const tonnes = data.predicted_yield_tonnes_per_hectare;
    const inputs = data.inputs;
    const ctx = data.context;

    $("#empty-state").hidden = true;
    const result = $("#result");
    result.hidden = false;
    result.classList.remove("reveal");
    void result.offsetWidth; // restart the animation
    result.classList.add("reveal");

    countUp($("#result-number"), tonnes);

    $("#result-meaning").textContent =
      `The model estimates approximately ${nf(0)(data.predicted_yield_kg_per_hectare)} kg of ${inputs.crop} per hectare.`;

    // Where the estimate sits among recorded yields for this crop
    $("#scale-low").textContent = `Lowest ${nf(2)(ctx.lowest)}`;
    $("#scale-median").textContent = `Median ${nf(2)(ctx.median)}`;
    $("#scale-high").textContent = `Highest ${nf(2)(ctx.highest)}`;
    $("#scale-text").textContent =
      `Higher than about ${Math.round(ctx.percentile)}% of the ${inputs.crop} yields recorded in the dataset (all countries and years, in t/ha).`;
    const marker = $("#scale-marker");
    marker.style.left = "0%";
    requestAnimationFrame(() => requestAnimationFrame(() => { marker.style.left = ctx.percentile + "%"; }));

    // Compare with the recorded value when the user loaded a real dataset record
    const compare = $("#compare");
    const example = state.activeExample;
    if (example && matchesExample(sent, example) && example.country === inputs.country && example.crop === inputs.crop) {
      const diff = tonnes - example.actual_t_ha;
      compare.replaceChildren();
      compare.append("Value recorded in the dataset for this record: ");
      const strong = document.createElement("strong");
      strong.textContent = `${nf(2)(example.actual_t_ha)} t/ha`;
      compare.append(strong, ` (the estimate is ${diff >= 0 ? "+" : "−"}${nf(2)(Math.abs(diff))} t/ha from it). This record was held out and never used to train the model.`);
      compare.hidden = false;
    } else {
      compare.hidden = true;
    }

    // Range warnings from the server
    const warnings = $("#warnings");
    warnings.replaceChildren();
    (data.warnings || []).forEach((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      warnings.append(li);
    });
    warnings.hidden = !(data.warnings && data.warnings.length);

    // The exact values that were used
    const rows = [
      ["Country", inputs.country],
      ["Crop", inputs.crop],
      ["Rainfall", `${nf(2)(inputs.rainfall)} mm`],
      ["Pesticides", `${nf(2)(inputs.pesticides)} tonnes`],
      ["Temperature", `${nf(2)(inputs.temperature)} °C`],
      ["Year", String(inputs.year)],
    ];
    const list = $("#inputs-used");
    list.replaceChildren();
    rows.forEach(([label, value]) => {
      const row = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label;
      dd.textContent = value;
      row.append(dt, dd);
      list.append(row);
    });

    if (window.matchMedia("(max-width: 960px)").matches) {
      $("#result-panel").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    }
  }

  function countUp(element, target) {
    if (reduceMotion) { element.textContent = target.toFixed(2); return; }
    const duration = 700;
    const start = performance.now();
    function frame(now) {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      element.textContent = (target * eased).toFixed(2);
      if (t < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  loadOptions();
})();
