import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {

  title = 'Bot Quijote';
  baseUrl: string = 'http://127.0.0.1:5000'
  seed: string;
  length: number;
  prediction: string;

  constructor(
    private httpClient: HttpClient
  ) { }

  getPrediction() {
    this.httpClient.get(this.baseUrl + '/predict/' + this.seed + '/' + this.length.toString()).subscribe((res) => {
      this.prediction = res.toString();
    });
  }

}
