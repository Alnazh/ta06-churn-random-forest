const form = document.querySelector("#form-prediksi");

if (form) {
  const toggleGroup = (selector, visible) => {
    form.querySelectorAll(selector).forEach((field) => {
      field.hidden = !visible;
    });
  };

  const syncDependents = () => {
    toggleGroup("[data-needs-internet]", form.elements.InternetService.value !== "No");
    toggleGroup("[data-needs-phone]", form.elements.PhoneService.value !== "No");
  };

  document.querySelectorAll("[data-fill]").forEach((button) => {
    button.addEventListener("click", () => {
      const values = JSON.parse(button.dataset.fill);
      Object.entries(values).forEach(([name, value]) => {
        if (form.elements[name]) form.elements[name].value = value;
      });
      syncDependents();
    });
  });

  form.elements.InternetService.addEventListener("change", syncDependents);
  form.elements.PhoneService.addEventListener("change", syncDependents);
  syncDependents();
}
