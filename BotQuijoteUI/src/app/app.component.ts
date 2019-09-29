import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {

  title = 'Bot Quijote';
  baseUrl: string = 'http://ec2-3-85-162-210.compute-1.amazonaws.com'
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
