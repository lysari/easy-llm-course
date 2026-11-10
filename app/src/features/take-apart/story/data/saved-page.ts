import html from "./weather-gov-lightning.json";

interface SavedPage {
  title: string;
  site: string;
  url: string;
  fetchedOn: string;
  licence: string;
  licenceUrl: string;
  html: string;
}

export const SAVED_PAGE: SavedPage = {
  title: "Understanding Lightning Science",
  site: "National Weather Service (weather.gov)",
  url: "https://www.weather.gov/safety/lightning-science-overview",
  fetchedOn: "2026-11-10",
  licence: "Public domain: National Weather Service pages are in the public domain unless noted otherwise.",
  licenceUrl: "https://www.weather.gov/disclaimer",
  html,
};
